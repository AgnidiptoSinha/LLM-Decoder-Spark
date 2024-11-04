package spark

import com.typesafe.config.ConfigFactory
import metrics.MetricsOutputHandler
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory
import utils.{InputTokens, S3Utils, TokenEmbedding}

import java.io.File
import java.time.Instant
import scala.util.{Failure, Success, Try}

object LLMTrainer {
  private val logger = LoggerFactory.getLogger(getClass)
  private val config = ConfigFactory.load()

  def main(args: Array[String]): Unit = {
    logger.info("Starting LLM Training Application")
    val startTime = Instant.now()

    // Load configuration
    val sparkMaster = config.getString("spark.master")
    val driverMemory = config.getString("spark.driverMemory")
    val executorMemory = config.getString("spark.executorMemory")
    val numWorkers = config.getInt("spark.numWorkers")
    val modelOutputDir = config.getString("model.outputDir")
    val metricsOutputDir = config.getString("model.metricsOutputDir")
    val embeddingsPath = config.getString("data.embeddingsPath")

    // Configure Spark with DL4J-specific settings
    val conf = new SparkConf()
      .setAppName("LLM Training")
      .setMaster(sparkMaster)
      .set("spark.driver.memory", driverMemory)
      .set("spark.executor.memory", executorMemory)

//      // Basic Kryo settings for ND4J
//      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
//      .set("spark.kryo.registrator", "org.nd4j.kryo.Nd4jRegistrator")

    val sc = new SparkContext(conf)

    try {
      // Initialize InputTokens with SparkContext
      val inputTokens = InputTokens(sc)

      // Load embeddings
      inputTokens.loadTokenEmbeddings(embeddingsPath) match {
        case Success(embeddings) if inputTokens.validateEmbeddings(embeddings) =>
          logger.info(s"Successfully loaded ${embeddings.size} token embeddings")
          trainModel(embeddings, sparkMaster, driverMemory, executorMemory, numWorkers, modelOutputDir, metricsOutputDir, inputTokens, sc)

        case Success(_) =>
          logger.error("Invalid embeddings found")
          sc.stop()
          System.exit(1)

        case Failure(e) =>
          logger.error(s"Failed to load embeddings: ${e.getMessage}")
          sc.stop()
          System.exit(1)
      }
    } catch {
      case e: Exception =>
        logger.error("Error during initialization", e)
        sc.stop()
        throw e
    }
  }

  private def trainModel(
                          embeddings: Vector[TokenEmbedding],
                          sparkMaster: String,
                          driverMemory: String,
                          executorMemory: String,
                          numWorkers: Int,
                          modelOutputDir: String,
                          metricsOutputDir: String,
                          inputTokens: InputTokens,
                          sc: SparkContext
                        ): Unit = {
    try {
      logger.info("Creating sliding window dataset")
      val windowSize = config.getInt("model.windowSize")
      val embeddingSize = inputTokens.getEmbeddingSize(embeddings)

      logger.info(s"Window size: $windowSize")
      logger.info(s"Embedding size: $embeddingSize")

      val numPartitions = calculateOptimalPartitions(sc.defaultParallelism)
      logger.info(s"Using $numPartitions partitions")

      // Convert embeddings to format needed by sliding window
      val tokens = embeddings.map(_.token.toString).toArray
//      val embeddingsMap = embeddings.map(e => e.token.toString -> e.embedding).toMap
      val embeddingsMap = embeddings.map { e =>
        e.token.toString -> e.embedding.toDoubleVector
      }.toMap

      val slidingWindowRDD = SparkSlidingWindowDataset.createSparkSlidingWindows(
        sc,
        tokens,
        windowSize,
        embeddingSize,
        numPartitions,
        Some(embeddingsMap)
      )

      // Split data into training and validation sets
      val splits = Array(
        config.getDouble("training.trainSplit"),
        config.getDouble("training.validSplit")
      )
      logger.info(s"Splitting data with ratios: ${splits.mkString(", ")}")

      val Array(trainData, validationData) = slidingWindowRDD.randomSplit(splits)

      // Log split sizes
      logger.info(s"Training set size: ${trainData.count()}")
      logger.info(s"Validation set size: ${validationData.count()}")

      // Create and train model
      logger.info("Initializing neural trainer")
      val trainer = SparkNeuralTrainer(sc)

      trainer.trainModel(trainData, Some(validationData)) match {
        case Success((model, metrics)) => {
          // Log final metrics
          logger.info("Training completed successfully")
          logger.info(s"Total training time: ${metrics.totalDurationMs / 1000.0} seconds")
          logger.info(s"Final training accuracy: ${metrics.epochs.last.accuracy}%")
          logger.info(s"Average batch processing time: ${metrics.averageBatchTimeMs}ms")
          logger.info(s"Peak memory usage: ${metrics.peakMemoryUsageMB}MB")

          // Save metrics to S3
          MetricsOutputHandler.saveMetricsToFile(metrics, metricsOutputDir, sc) match {
            case Success(filePath) =>
              logger.info(s"Training metrics saved to: $filePath")
            case Failure(e) =>
              logger.error("Failed to save metrics to file", e)
          }

          // Save the trained model to S3
          val modelStream = new java.io.ByteArrayOutputStream()
          ModelSerializer.writeModel(model, modelStream, true)
          val modelPath = s"$modelOutputDir/llm_model_${Instant.now().getEpochSecond}.zip"
          S3Utils.writeToS3(sc, modelStream.toByteArray, modelPath)
          logger.info(s"Model saved to $modelPath")
        }
        case Failure(e) => {
          logger.error("Training failed", e)
          throw e
        }
      }
    } catch {
      case e: Exception =>
        logger.error("Error during training", e)
        throw e
    }
  }

  private def calculateOptimalPartitions(defaultParallelism: Int): Int = {
    val minPartitions = 1
    val maxPartitions = defaultParallelism * 2
    val targetPartitionSize = config.getInt("spark.targetPartitionSize")

    Math.min(maxPartitions, Math.max(minPartitions, defaultParallelism))
  }
}