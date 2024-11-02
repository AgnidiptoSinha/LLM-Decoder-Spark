package utils

import org.apache.spark.rdd.RDD
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.Logger

case class MemoryStats(
                        memoryPerPartition: Double,
                        totalMemory: Double
                      )

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

  // Log detailed RDD statistics
  def logRDDStats(rdd: RDD[DataSet], logger: Logger): Unit = {
    logger.info(s"Number of partitions: ${rdd.getNumPartitions}")
    val count = rdd.count()
    logger.info(s"Total number of elements: $count")

    // Get partition sizes
    val partitionSizes = rdd.mapPartitionsWithIndex { (idx, iter) =>
      Iterator((idx, iter.size))
    }.collect()

    logger.info("Partition distribution:")
    partitionSizes.foreach { case (idx, size) =>
      logger.info(s"Partition $idx: $size elements")

      // Calculate and log partition balance
      val percentage = (size.toDouble / count) * 100
      logger.info(f"Partition $idx load: $percentage%.2f%%")
    }

    // Calculate partition balance metrics
    val sizes = partitionSizes.map(_._2)
    val avgSize = sizes.sum.toDouble / sizes.length
    val stdDev = Math.sqrt(sizes.map(s => Math.pow(s - avgSize, 2)).sum / sizes.length)

    logger.info(f"Average partition size: $avgSize%.2f")
    logger.info(f"Partition size standard deviation: $stdDev%.2f")
  }

  // Calculate memory statistics
  def calculateMemoryStats(rdd: RDD[DataSet]): MemoryStats = {
    // Sample a few DataSets to estimate memory usage
    val sampleSize = Math.min(rdd.count().toInt, 10)
    val samples = rdd.take(sampleSize)

    // Estimate memory usage per DataSet (rough approximation)
    val avgMemoryPerDataSet = samples.map { ds =>
      val features = ds.getFeatures
      val labels = ds.getLabels
      (features.length() + labels.length()) * 4 // 4 bytes per float
    }.sum.toDouble / sampleSize

    val numPartitions = rdd.getNumPartitions
    val elementsPerPartition = rdd.count().toDouble / numPartitions

    val memoryPerPartition = (avgMemoryPerDataSet * elementsPerPartition) / (1024 * 1024) // Convert to MB
    val totalMemory = memoryPerPartition * numPartitions

    MemoryStats(memoryPerPartition, totalMemory)
  }
}