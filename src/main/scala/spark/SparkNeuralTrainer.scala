package spark

import com.typesafe.config.ConfigFactory
import metrics._
import org.apache.spark.SparkContext
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.slf4j.LoggerFactory

import java.time.{Duration, Instant}
import scala.collection.JavaConverters._
import scala.util.{Try, Success, Failure}

case class DataPreprocessor(inputSize: Int, outputSize: Int) extends Serializable {
  def preprocess(dataset: DataSet): DataSet = {
    // Ensure input features are shaped correctly
    val features = dataset.getFeatures
    val labels = dataset.getLabels

    // Reshape features if needed
    val reshapedFeatures = if (features.shape()(1) != inputSize) {
      features.reshape(1, inputSize)
    } else {
      features
    }

    // Reshape labels if needed
    val reshapedLabels = if (labels.shape()(1) != outputSize) {
      labels.reshape(1, outputSize)
    } else {
      labels
    }

    // Create new dataset with reshaped data
    new DataSet(reshapedFeatures, reshapedLabels)
  }
}

class SparkNeuralTrainer(sc: SparkContext) extends Serializable {
  @transient private lazy val logger = LoggerFactory.getLogger(getClass)
  private val metricsCollector = new TrainingMetricsCollector()

  private val config = ConfigFactory.load()
  private val batchSize = config.getInt("training.batchSize")
  private val numEpochs = config.getInt("training.numEpochs")
  private val learningRate = config.getDouble("training.learningRate")
  private val windowSize = config.getInt("model.windowSize")
  private val embeddingSize = config.getInt("model.embeddingSize")
  private val hiddenSize = config.getInt("model.hiddenSize")

  private val inputSize = windowSize * embeddingSize
  private val outputSize = embeddingSize

  private def safelyGetGradientStats(network: MultiLayerNetwork): GradientStats = {
    Try {
      Option(network)
        .flatMap(n => Option(n.gradient()))
        .flatMap(g => Option(g.gradient()))
        .map { gradientArray =>
          GradientStats(
            norm = gradientArray.norm2Number().doubleValue(),
            min = gradientArray.minNumber().doubleValue(),
            max = gradientArray.maxNumber().doubleValue(),
            mean = gradientArray.meanNumber().doubleValue(),
            standardDeviation = Math.sqrt(gradientArray.varNumber().doubleValue())
          )
        }.getOrElse(GradientStats(0.0, 0.0, 0.0, 0.0, 0.0))
    } match {
      case Success(stats) => stats
      case Failure(e) =>
        logger.warn(s"Failed to collect gradient statistics: ${e.getMessage}")
        GradientStats(0.0, 0.0, 0.0, 0.0, 0.0)
    }
  }

  private def getCurrentLearningRate(network: MultiLayerNetwork): Double = {
    Try {
      Option(network)
        .flatMap(n => Option(n.conf()))
        .flatMap(c => Option(c.getLayer))
        .flatMap(l => Option(l.getUpdaterByParam("W")))
        .map(_.asInstanceOf[Adam].getLearningRate)
        .getOrElse(learningRate)
    } match {
      case Success(rate) => rate
      case Failure(e) =>
        logger.warn(s"Failed to get current learning rate: ${e.getMessage}")
        learningRate
    }
  }

  private def collectBatchMetrics(
                                   network: MultiLayerNetwork,
                                   batchNum: Int,
                                   loss: Double,
                                   batchStartTime: Long
                                 ): BatchMetrics = {
    try {
      val currentTime = System.currentTimeMillis()
      val memoryUsed = (Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()) / (1024.0 * 1024.0)

      BatchMetrics(
        batchNumber = batchNum,
        loss = loss,
        accuracy = calculateAccuracy(network, sc.emptyRDD[DataSet]),
        gradientStats = safelyGetGradientStats(network),
        learningRate = getCurrentLearningRate(network),
        processingTimeMs = currentTime - batchStartTime,
        memoryUsageMB = memoryUsed
      )
    } catch {
      case e: Exception =>
        logger.error(s"Error collecting batch metrics: ${e.getMessage}")
        // Return default metrics if collection fails
        BatchMetrics(
          batchNumber = batchNum,
          loss = Double.NaN,
          accuracy = 0.0,
          gradientStats = GradientStats(0.0, 0.0, 0.0, 0.0, 0.0),
          learningRate = learningRate,
          processingTimeMs = 0L,
          memoryUsageMB = 0.0
        )
    }
  }

  private def calculateAccuracy(network: MultiLayerNetwork, data: RDD[DataSet]): Double = {
    Try {
      if (data.isEmpty()) {
        0.0
      } else {
        val sampleSize = Math.min(1000, data.count().toInt)
        val sampleData = data.take(sampleSize)

        val correct = sampleData.count { dataset =>
          if (dataset != null && dataset.getFeatures != null && dataset.getLabels != null) {
            val predicted = network.output(dataset.getFeatures)
            val actual = dataset.getLabels

            // Calculate cosine similarity as accuracy metric for embeddings
            val similarity = calculateCosineSimilarity(predicted, actual)
            similarity > 0.5 // Consider predictions with >10% similarity as correct
          } else {
            false
          }
        }

        correct.toDouble / sampleSize * 100 // Return percentage
      }
    } match {
      case Success(acc) => acc
      case Failure(e) =>
        logger.warn(s"Failed to calculate accuracy: ${e.getMessage}")
        0.0
    }
  }

  private def calculateCosineSimilarity(predicted: INDArray, actual: INDArray): Double = {
    Try {
      if (predicted == null || actual == null) return 0.0

      val dotProduct = predicted.mul(actual).sumNumber().doubleValue()
      val normPredicted = predicted.norm2Number().doubleValue()
      val normActual = actual.norm2Number().doubleValue()

      if (normPredicted > 0 && normActual > 0) {
        dotProduct / (normPredicted * normActual)
      } else {
        0.0
      }
    } match {
      case Success(sim) => sim
      case Failure(e) =>
        logger.warn(s"Failed to calculate cosine similarity: ${e.getMessage}")
        0.0
    }
  }

  private def calculateScore(sparkNet: SparkDl4jMultiLayer, data: RDD[DataSet]): Double = {
    Try {
      if (data.isEmpty()) {
        Double.NaN
      } else {
        val sampleSize = Math.min(1000, data.count().toInt)
        val sampleData = data.take(sampleSize)

        val combinedData = new DataSet(
          org.nd4j.linalg.factory.Nd4j.vstack(sampleData.map(_.getFeatures).toList.asJava),
          org.nd4j.linalg.factory.Nd4j.vstack(sampleData.map(_.getLabels).toList.asJava)
        )

        sparkNet.getNetwork.score(combinedData)
      }
    } match {
      case Success(score) => score
      case Failure(e) =>
        logger.warn(s"Failed to calculate score: ${e.getMessage}")
        Double.NaN
    }
  }

  def createNetwork(): MultiLayerNetwork = {
    logger.info(s"Creating neural network with input size: $inputSize, hidden size: $hiddenSize, output size: $outputSize")

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

  private def collectEpochMetrics(
                                   sparkNet: SparkDl4jMultiLayer,
                                   trainData: RDD[DataSet],
                                   validationData: Option[RDD[DataSet]],
                                   epochNum: Int,
                                   epochStartTime: Instant,
                                   batchMetrics: Vector[BatchMetrics]
                                 ): EpochMetrics = {
    try {
      logger.info(s"Collecting metrics for epoch $epochNum")

      val network = Option(sparkNet).map(_.getNetwork).orNull
      if (network == null) {
        logger.warn("Network is null, returning default metrics")
        return createDefaultEpochMetrics(epochNum, epochStartTime, batchMetrics)
      }

      // Safely calculate training metrics
      val (loss, accuracy) = Try {
        val l = calculateScore(sparkNet, trainData)
        val a = calculateAccuracy(network, trainData)
        (l, a)
      }.getOrElse((Double.NaN, 0.0))

      // Safely calculate validation metrics
      val validationMetrics = validationData.flatMap { valData =>
        Try {
          val valLoss = calculateScore(sparkNet, valData)
          val valAccuracy = calculateAccuracy(network, valData)
          Some((valLoss, valAccuracy))
        }.getOrElse(None)
      }

      // Safely get gradient statistics
      val gradStats = safelyGetGradientStats(network)

      // Safely get learning rate
      val currentLearningRate = getCurrentLearningRate(network)

      EpochMetrics(
        epochNumber = epochNum,
        loss = loss,
        accuracy = accuracy,
        validationLoss = validationMetrics.map(_._1),
        validationAccuracy = validationMetrics.map(_._2),
        gradientStats = gradStats,
        learningRate = currentLearningRate,
        batchMetrics = batchMetrics,
        durationMs = Duration.between(epochStartTime, Instant.now()).toMillis,
        timestamp = Instant.now()
      )
    } catch {
      case e: Exception =>
        logger.error(s"Error collecting epoch metrics: ${e.getMessage}", e)
        createDefaultEpochMetrics(epochNum, epochStartTime, batchMetrics)
    }
  }

  private def createDefaultEpochMetrics(
                                         epochNum: Int,
                                         epochStartTime: Instant,
                                         batchMetrics: Vector[BatchMetrics]
                                       ): EpochMetrics = {
    EpochMetrics(
      epochNumber = epochNum,
      loss = Double.NaN,
      accuracy = 0.0,
      validationLoss = None,
      validationAccuracy = None,
      gradientStats = GradientStats(0.0, 0.0, 0.0, 0.0, 0.0),
      learningRate = learningRate,
      batchMetrics = batchMetrics,
      durationMs = Duration.between(epochStartTime, Instant.now()).toMillis,
      timestamp = Instant.now()
    )
  }

  def trainModel(
                  trainData: RDD[DataSet],
                  validationData: Option[RDD[DataSet]] = None
                ): Try[(MultiLayerNetwork, TrainingMetrics)] = {
    Try {
      logger.info("Starting model training")
      metricsCollector.startTraining()

      val preprocessor = DataPreprocessor(inputSize, outputSize)
      val processedTrainData = trainData.map(preprocessor.preprocess)

      val network = createNetwork()
      if (network == null) {
        throw new IllegalStateException("Failed to create neural network")
      }

      val totalParams = network.numParams()
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

      var totalBatchTime = 0L
      var batchesProcessed = 0L
      val epochMetricsBuffer = new scala.collection.mutable.ArrayBuffer[EpochMetrics]()

      // Training loop
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
          val batchMetrics = collectBatchMetrics(
            sparkNet.getNetwork,
            batchesProcessed.toInt,
            calculateScore(sparkNet, processedTrainData),
            batchStartTime
          )
          batchMetricsBuffer += batchMetrics
          metricsCollector.logBatchMetrics(batchMetrics)
        }.recover {
          case e: Exception =>
            logger.error(s"Error in training iteration for epoch $epoch: ${e.getMessage}", e)
        }

        // Collect epoch metrics
        val epochMetrics = collectEpochMetrics(
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