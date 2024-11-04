package spark

import com.typesafe.config.ConfigFactory
import metrics._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import utils.{DataPreprocessor, NeuralTrainerUtils}

import java.time.{Duration, Instant}
import scala.util.Try

class SparkNeuralTrainer(sc: SparkContext) extends Serializable {
  @transient private lazy val logger = LoggerFactory.getLogger(getClass)
  private val metricsCollector = new TrainingMetricsCollector()
  private val trainerUtils = new NeuralTrainerUtils(sc)

  private val config = ConfigFactory.load()
  private val batchSize = config.getInt("training.batchSize")
  private val numEpochs = config.getInt("training.numEpochs")
  private val learningRate = config.getDouble("training.learningRate")
  private val windowSize = config.getInt("model.windowSize")
  private val embeddingSize = config.getInt("model.embeddingSize")
  private val hiddenSize = config.getInt("model.hiddenSize")

  private val inputSize = windowSize * embeddingSize
  private val outputSize = embeddingSize

  logger.info(s"Initializing trainer with:")
  logger.info(s"Window size: $windowSize")
  logger.info(s"Embedding size: $embeddingSize")
  logger.info(s"Input size: $inputSize")
  logger.info(s"Output size: $outputSize")

  private def createNetwork(): MultiLayerNetwork = {
    logger.info(s"Creating neural network with input size: $inputSize, hidden size: $hiddenSize, output size: $outputSize")

    // Validate dimensions before creating the network
    trainerUtils.validateModelDimensions()

    logger.info(s"Creating neural network with validated dimensions:")
    logger.info(s"Input size: $inputSize")
    logger.info(s"Hidden size: $hiddenSize")
    logger.info(s"Output size: $outputSize")

    val conf = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam(learningRate))
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(inputSize)
        .nOut(hiddenSize)
        .activation(Activation.RELU)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(hiddenSize)
        .nOut(hiddenSize)
        .activation(Activation.RELU)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .nIn(hiddenSize)
        .nOut(outputSize)
        .activation(Activation.IDENTITY)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(metricsCollector)
    model
  }

  def trainModel(
                  trainData: RDD[DataSet],
                  validationData: Option[RDD[DataSet]] = None
                ): Try[(MultiLayerNetwork, TrainingMetrics)] = {
    Try {
      logger.info("Starting model training")

      // Validate dimensions before training
      trainerUtils.validateModelDimensions()

      metricsCollector.startTraining()

      val preprocessor = DataPreprocessor(inputSize, outputSize)
      val processedTrainData = trainData.map(preprocessor.preprocess)

      val network = createNetwork()
      if (network == null) {
        throw new IllegalStateException("Failed to create neural network")
      }

      // Mutable counters required to track cumulative statistics across training iterations
      var totalParams = network.numParams()  // Counter for total network parameters that may change during training
      logger.info(s"Created network with $totalParams parameters")

      val tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
        .workerPrefetchNumBatches(2)
        .batchSizePerWorker(batchSize)
        .averagingFrequency(1)
        .storageLevel(StorageLevel.MEMORY_AND_DISK_SER)
        .rddTrainingApproach(org.deeplearning4j.spark.api.RDDTrainingApproach.Direct)
        .build()

      val sparkNet = new SparkDl4jMultiLayer(sc, network, tm)
      if (sparkNet == null) {
        throw new IllegalStateException("Failed to create Spark neural network")
      }

      // Mutable counters required to track cumulative statistics across training iterations
      var totalBatchTime = 0L        // Accumulator for total processing time across all batches
      var batchesProcessed = 0L      // Counter for total number of batches processed
      val epochMetricsBuffer = new scala.collection.mutable.ArrayBuffer[EpochMetrics]()

      // Training loop
      // Using traditional for loop for clarity in stateful training process
      // Could use (1 to numEpochs).foreach but imperative style better expresses training loop semantics
      for (epoch <- 1 to numEpochs) {
        val epochStartTime = Instant.now()
        logger.info(s"Starting epoch $epoch")

        val batchMetricsBuffer = new scala.collection.mutable.ArrayBuffer[BatchMetrics]()

        Try {
          // Fit the model
          val batchStartTime = System.currentTimeMillis()
          sparkNet.fit(processedTrainData)
          val batchTime = System.currentTimeMillis() - batchStartTime

          totalBatchTime += batchTime
          batchesProcessed += processedTrainData.count()

          // Collect batch metrics
          val batchMetrics = trainerUtils.collectBatchMetrics(
            sparkNet.getNetwork,
            batchesProcessed.toInt,
            trainerUtils.calculateScore(sparkNet, processedTrainData),
            batchStartTime
          )
          batchMetricsBuffer += batchMetrics
          metricsCollector.logBatchMetrics(batchMetrics)
        }.recover {
          case e: Exception =>
            logger.error(s"Error in training iteration for epoch $epoch: ${e.getMessage}", e)
        }

        // Collect epoch metrics
        val epochMetrics = trainerUtils.collectEpochMetrics(
          sparkNet,
          processedTrainData,
          validationData,
          epoch,
          epochStartTime,
          batchMetricsBuffer.toVector
        )
        epochMetricsBuffer += epochMetrics
        metricsCollector.logEpochMetrics(epochMetrics)
      }

      metricsCollector.endTraining()

      // Create final metrics
      val finalMetrics = TrainingMetrics(
        epochs = epochMetricsBuffer.toVector,
        totalDurationMs = Duration.between(
          epochMetricsBuffer.head.timestamp,
          epochMetricsBuffer.last.timestamp
        ).toMillis,
        averageBatchTimeMs = if (batchesProcessed > 0) totalBatchTime.toDouble / batchesProcessed else 0.0,
        peakMemoryUsageMB = if (epochMetricsBuffer.isEmpty) 0.0
        else epochMetricsBuffer.flatMap(_.batchMetrics.map(_.memoryUsageMB)).foldLeft(0.0)(Math.max),
        sparkMetrics = metricsCollector.collectSparkMetrics(sc, processedTrainData),
        modelMetrics = ModelMetrics(
          totalParameters = totalParams,
          layerSizes = Vector(inputSize, hiddenSize, hiddenSize, outputSize),
          batchesProcessed = batchesProcessed,
          averageProcessingTimePerBatchMs = if (batchesProcessed > 0) totalBatchTime.toDouble / batchesProcessed else 0.0
        )
      )

      metricsCollector.logFinalMetrics(finalMetrics)

      (sparkNet.getNetwork, finalMetrics)
    }
  }
}

object SparkNeuralTrainer {
  def apply(sc: SparkContext): SparkNeuralTrainer = new SparkNeuralTrainer(sc)
}