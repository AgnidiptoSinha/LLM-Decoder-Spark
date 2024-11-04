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
class DatasetCreator(
                      embeddingSize: Int,
                     existingEmbeddings: Option[Map[String, Array[Double]]] = None
                    ) extends Serializable {

  def createEmbeddings(tokens: Array[String]): Map[String, INDArray] = {
    existingEmbeddings.getOrElse {
      tokens.map(token => token -> Nd4j.rand(embeddingSize).toDoubleVector).toMap
    }.map { case (token, array) =>
      token -> Nd4j.create(array)
    }
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
                                 numPartitions: Int,
                                 existingEmbeddings: Option[Map[String, Array[Double]]] = None
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

    // Create DatasetCreator instance with existing embeddings if provided
    val creator = new DatasetCreator(embeddingSize, existingEmbeddings)

    // Process each window to create DataSet
    windowsRDD.map(creator.createDataSet)
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