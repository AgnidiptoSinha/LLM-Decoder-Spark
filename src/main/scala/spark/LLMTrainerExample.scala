package spark

import com.typesafe.config.ConfigFactory
import org.apache.spark.{SparkConf, SparkContext}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory
import java.io.File
import java.time.Instant

import scala.util.{Failure, Success, Try}

object LLMTrainerExample {
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

    // Configure Spark with DL4J-specific settings
    val conf = new SparkConf()
      .setAppName("LLM Training")
      .setMaster(sparkMaster)
      // Memory settings
      .set("spark.driver.memory", driverMemory)
      .set("spark.executor.memory", executorMemory)
      // ND4J Kryo configuration
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryo.registrator", "org.nd4j.kryo.Nd4jRegistrator")
      // Kryo buffer settings
      .set("spark.kryoserializer.buffer.max", "2047m")
      .set("spark.kryoserializer.buffer", "512m")
      // Performance settings
      .set("spark.memory.fraction", "0.6")
      .set("spark.memory.storageFraction", "0.5")
      .set("spark.network.timeout", "800s")
      .set("spark.executor.heartbeatInterval", "20s")
      // Performance tuning
      .set("spark.rdd.compress", "true")
      .set("spark.broadcast.compress", "true")
      .set("spark.shuffle.compress", "true")
      // Disable dynamic allocation for consistent performance
      .set("spark.dynamicAllocation.enabled", "false")
      // Set fixed number of cores
      .set("spark.cores.max", numWorkers.toString)
      // Additional DL4J specific settings
      .set("spark.task.maxFailures", "1")
      .set("spark.locality.wait", "0s")
      .set("spark.submit.deployMode", "client")

    val sc = new SparkContext(conf)

    try {
      logger.info("Initializing training data")

      // Sample data - replace with your actual data loading logic from homework 1
      val tokens = Array("The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog")

      // Create sliding window dataset
      logger.info("Creating sliding window dataset")
      logger.info(s"Window size: ${config.getInt("model.windowSize")}")
      logger.info(s"Embedding size: ${config.getInt("model.embeddingSize")}")

      val numPartitions = calculateOptimalPartitions(sc.defaultParallelism)
      logger.info(s"Using $numPartitions partitions")

      val slidingWindowRDD = SparkSlidingWindowDataset.createSparkSlidingWindows(
        sc,
        tokens,
        config.getInt("model.windowSize"),
        config.getInt("model.embeddingSize"),
        numPartitions
      )

      // Cache the RDD with storage level specification
//      slidingWindowRDD.persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER)

      // Log initial dataset statistics
      val datasetSize = slidingWindowRDD.count()
      logger.info(s"Created dataset with $datasetSize samples")
      logger.info(s"Number of partitions: ${slidingWindowRDD.getNumPartitions}")

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

          // Save the trained model
          new File(modelOutputDir).mkdirs()
          val modelFile = new File(s"$modelOutputDir/llm_model_${Instant.now().getEpochSecond}.zip")
          ModelSerializer.writeModel(model, modelFile, true)
          logger.info(s"Model saved to ${modelFile.getAbsolutePath}")
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
    } finally {
      logger.info("Cleaning up resources")
      sc.stop()

      val endTime = Instant.now()
      val totalTime = java.time.Duration.between(startTime, endTime).getSeconds
      logger.info(s"Total application runtime: ${totalTime}s")
    }
  }

  private def calculateOptimalPartitions(defaultParallelism: Int): Int = {
    val minPartitions = 1
    val maxPartitions = defaultParallelism * 2
    val targetPartitionSize = config.getInt("spark.targetPartitionSize")

    Math.min(maxPartitions, Math.max(minPartitions, defaultParallelism))
  }
}