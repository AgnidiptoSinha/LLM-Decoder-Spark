package spark

import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

// Case class for window data
case class WindowData(
                       inputTokens: Array[String],
                       targetToken: String
                     ) extends Serializable

// Helper class for creating embeddings and datasets
class DatasetCreator(embeddingSize: Int) extends Serializable {
  def createEmbeddings(tokens: Array[String]): Map[String, INDArray] = {
    tokens.map(token => token -> Nd4j.rand(embeddingSize)).toMap
  }

  def computePositionalEmbedding(position: Int): INDArray = {
    val posEmbedding = Nd4j.zeros(embeddingSize)

    for (i <- 0 until embeddingSize by 2) {
      val angle = position / Math.pow(10000, (2.0 * i) / embeddingSize)
      posEmbedding.putScalar(i, Math.sin(angle))
      posEmbedding.putScalar(i + 1, Math.cos(angle))
    }

    posEmbedding
  }

  def createDataSet(window: WindowData): DataSet = {
    val embeddings = createEmbeddings(window.inputTokens ++ Array(window.targetToken))
    val inputEmbeddings = combineEmbeddings(window.inputTokens, embeddings)
    val targetEmbedding = embeddings(window.targetToken)

    val input = inputEmbeddings.reshape(1, embeddingSize * window.inputTokens.length)
    val target = targetEmbedding.reshape(1, embeddingSize)

    new DataSet(input, target)
  }

  private def combineEmbeddings(
                                 window: Array[String],
                                 embeddings: Map[String, INDArray]
                               ): INDArray = {
    val combinedEmbedding = Nd4j.zeros(window.length, embeddingSize)

    for (i <- window.indices) {
      val embedding = embeddings(window(i))
      val positionalEmbedding = computePositionalEmbedding(i)
      combinedEmbedding.putRow(i, embedding.add(positionalEmbedding))
    }

    combinedEmbedding
  }
}

object SparkSlidingWindowDataset {
  private val logger = LoggerFactory.getLogger(getClass)

  def createSparkSlidingWindows(
                                 sc: SparkContext,
                                 tokens: Array[String],
                                 windowSize: Int,
                                 embeddingSize: Int,
                                 numPartitions: Int
                               ): RDD[DataSet] = {

    // Create windows of tokens
    val windows = (0 until tokens.length - windowSize).map { startIndex =>
      WindowData(
        tokens.slice(startIndex, startIndex + windowSize),
        tokens(startIndex + windowSize)
      )
    }

    // Create RDD from windows
    val windowsRDD = sc.parallelize(windows, numPartitions)

    // Create DatasetCreator instance
    val creator = new DatasetCreator(embeddingSize)

    // Process each window to create DataSet
    windowsRDD.map(creator.createDataSet)
  }
}

object SparkSlidingWindowExample {
  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    // Spark Configuration
    val conf = new SparkConf()
      .setAppName("Spark Sliding Window Dataset")
      .setMaster("local[*]")
      .set("spark.driver.memory", "4g")
      .set("spark.executor.memory", "4g")

    val sc = new SparkContext(conf)

    try {
      // Sample data
      val sampleTokens = Array("The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog")
      val windowSize = 4
      val embeddingSize = 128

      // Calculate optimal partitions
      val numWindows = sampleTokens.length - windowSize
      val numPartitions = SparkSlidingWindowUtils.calculateOptimalPartitions(numWindows)
      logger.info(s"Using $numPartitions partitions")

      // Create and process the RDD
      val slidingWindowRDD = SparkSlidingWindowDataset.createSparkSlidingWindows(
        sc,
        sampleTokens,
        windowSize,
        embeddingSize,
        numPartitions
      )

      // Cache the RDD
      slidingWindowRDD.cache()

      // Log RDD statistics
      SparkSlidingWindowUtils.logRDDStats(slidingWindowRDD, logger)

      // Process sample windows
      val firstFewDataSets = slidingWindowRDD.take(3)
      logger.info(s"No. of Datasets: ${slidingWindowRDD.count()}")
      firstFewDataSets.zipWithIndex.foreach { case (ds, idx) =>
        logger.info(s"Window $idx:")
        logger.info(s"Input shape: ${ds.getFeatures.shape().mkString("x")}")
        logger.info(s"Target shape: ${ds.getLabels.shape().mkString("x")}")
      }

    } catch {
      case e: Exception =>
        logger.error("Error in Spark sliding window creation", e)
        throw e
    } finally {
      sc.stop()
    }
  }
}

object SparkSlidingWindowUtils {
  // Calculate optimal number of partitions based on data size
  def calculateOptimalPartitions(
                                  dataSize: Int,
                                  targetPartitionSize: Int = 1000
                                ): Int = {
    val minPartitions = 1
    val maxPartitions = Runtime.getRuntime.availableProcessors() * 2
    val calculatedPartitions = Math.max(1, dataSize / targetPartitionSize)

    Math.min(calculatedPartitions, maxPartitions)
  }

  // Log RDD statistics
  def logRDDStats(rdd: RDD[DataSet], logger: org.slf4j.Logger): Unit = {
    logger.info(s"Number of partitions: ${rdd.getNumPartitions}")
    val count = rdd.count()
    logger.info(s"Total number of elements: $count")

    val partitionSizes = rdd.mapPartitionsWithIndex { (idx, iter) =>
      Iterator((idx, iter.size))
    }.collect()

    logger.info("Partition distribution:")
    partitionSizes.foreach { case (idx, size) =>
      logger.info(s"Partition $idx: $size elements")
    }
  }
}