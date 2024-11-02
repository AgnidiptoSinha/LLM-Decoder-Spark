package metrics

import org.apache.spark.SparkContext
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory

import java.time.{Duration, Instant}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd.RDD

import org.deeplearning4j.nn.api.Model

// Core metrics case classes
case class GradientStats(
                          norm: Double,
                          min: Double,
                          max: Double,
                          mean: Double,
                          standardDeviation: Double
                        )

case class BatchMetrics(
                         batchNumber: Int,
                         loss: Double,
                         accuracy: Double,
                         gradientStats: GradientStats,
                         learningRate: Double,
                         processingTimeMs: Long,
                         memoryUsageMB: Double
                       )

case class EpochMetrics(
                         epochNumber: Int,
                         loss: Double,
                         accuracy: Double,
                         validationLoss: Option[Double],
                         validationAccuracy: Option[Double],
                         gradientStats: GradientStats,
                         learningRate: Double,
                         batchMetrics: Vector[BatchMetrics],
                         durationMs: Long,
                         timestamp: Instant
                       )

case class SparkExecutorMetrics(
                                 host: String,
                                 port: Int,
                                 usedStorageMemoryMB: Double,
                                 totalStorageMemoryMB: Double,
                                 numRunningTasks: Int
                               )

case class TrainingMetrics(
                            epochs: Vector[EpochMetrics],
                            totalDurationMs: Long,
                            averageBatchTimeMs: Double,
                            peakMemoryUsageMB: Double,
                            sparkMetrics: SparkTrainingMetrics,
                            modelMetrics: ModelMetrics
                          )

case class SparkTrainingMetrics(
                                 executorMetrics: Vector[SparkExecutorMetrics],
                                 dataDistributionStats: DataDistributionStats,
                                 activeExecutors: Int,
                                 totalExecutorMemory: Double, // in MB
                                 activeJobCount: Int
                               )

case class ModelMetrics(
                         totalParameters: Long,
                         layerSizes: Vector[Int],
                         batchesProcessed: Long,
                         averageProcessingTimePerBatchMs: Double
                       )

case class DataDistributionStats(
                                  totalPartitions: Int,
                                  recordsPerPartition: Map[Int, Long],
                                  partitionSkewPercentage: Double
                                )

class TrainingMetricsCollector extends BaseTrainingListener with Serializable {
  private val logger = LoggerFactory.getLogger(getClass)
  private val epochMetrics = ArrayBuffer[EpochMetrics]()
  private val currentBatchMetrics = ArrayBuffer[BatchMetrics]()
  private var startTime: Instant = _
  private var endTime: Instant = _
  private var peakMemory: Double = 0.0

  def startTraining(): Unit = {
    startTime = Instant.now()
    logger.info("Started training metrics collection")
  }

  def endTraining(): Unit = {
    endTime = Instant.now()
    val duration = Duration.between(startTime, endTime)
    logger.info(f"Training completed in ${duration.toMinutes}%d minutes")
  }

  // Properly override BaseTrainingListener methods
  override def onForwardPass(model: Model, activations: java.util.List[INDArray]): Unit = {
    logger.debug(s"Forward pass completed with ${activations.size()} layer activations")
  }

  override def onForwardPass(model: Model, activations: java.util.Map[String, INDArray]): Unit = {
    logger.debug(s"Forward pass completed with ${activations.size()} named activations")
  }

  override def onBackwardPass(model: Model): Unit = {
    // Handle gradient statistics
    if (model.gradient() != null && model.gradient().gradient() != null) {
      val gradientArray = model.gradient().gradient()
      val gradStats = GradientStats(
        norm = gradientArray.norm2Number().doubleValue(),
        min = gradientArray.minNumber().doubleValue(),
        max = gradientArray.maxNumber().doubleValue(),
        mean = gradientArray.meanNumber().doubleValue(),
        standardDeviation = calculateStdDev(gradientArray)
      )
      logGradientStats(gradStats)
    }
  }

  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    // Record iteration metrics
    logger.debug(s"Completed iteration $iteration in epoch $epoch")
  }

  private def calculateStdDev(array: INDArray): Double = {
    val mean = array.meanNumber().doubleValue()
    val variance = array.sub(mean).muli(array.sub(mean)).meanNumber().doubleValue()
    Math.sqrt(variance)
  }

  def collectSparkMetrics(sc: SparkContext, data: RDD[DataSet]): SparkTrainingMetrics = {
    // Get executor metrics
    val executorMetrics = sc.statusTracker.getExecutorInfos.map { executor =>
      SparkExecutorMetrics(
        host = executor.host(),
        port = executor.port(),
        usedStorageMemoryMB = (executor.usedOnHeapStorageMemory() +
          executor.usedOffHeapStorageMemory()) / (1024.0 * 1024.0),
        totalStorageMemoryMB = (executor.totalOnHeapStorageMemory() +
          executor.totalOffHeapStorageMemory()) / (1024.0 * 1024.0),
        numRunningTasks = executor.numRunningTasks()
      )
    }.toVector

    // Calculate total memory and active executors
    val totalMemory = executorMetrics.map(_.totalStorageMemoryMB).sum
    val activeExecutors = executorMetrics.size

    // Get partition statistics
    val partitionStats = calculatePartitionStats(data)

    SparkTrainingMetrics(
      executorMetrics = executorMetrics,
      dataDistributionStats = partitionStats,
      activeExecutors = activeExecutors,
      totalExecutorMemory = totalMemory,
      activeJobCount = sc.statusTracker.getActiveJobIds().length
    )
  }

  // Rest of the methods remain the same...

  private def calculatePartitionStats(data: RDD[DataSet]): DataDistributionStats = {
    val partitionCounts = data.mapPartitionsWithIndex { (idx, iter) =>
      Iterator((idx, iter.size.toLong))
    }.collectAsMap()

    val totalRecords = partitionCounts.values.sum
    val avgRecordsPerPartition = totalRecords.toDouble / partitionCounts.size
    val maxSkew = partitionCounts.values.map(count =>
      Math.abs(count - avgRecordsPerPartition) / avgRecordsPerPartition * 100
    ).max

    DataDistributionStats(
      totalPartitions = partitionCounts.size,
      recordsPerPartition = partitionCounts.toMap,
      partitionSkewPercentage = maxSkew
    )
  }

  private def logGradientStats(stats: GradientStats): Unit = {
    logger.info(
      f"""Gradient Statistics:
         |  Norm: ${stats.norm}%.6f
         |  Min: ${stats.min}%.6f
         |  Max: ${stats.max}%.6f
         |  Mean: ${stats.mean}%.6f
         |  Std Dev: ${stats.standardDeviation}%.6f""".stripMargin)
  }

  def logBatchMetrics(metrics: BatchMetrics): Unit = {
    logger.info(
      f"""Batch ${metrics.batchNumber}%d:
         |  Loss: ${metrics.loss}%.6f
         |  Accuracy: ${metrics.accuracy}%.2f%%
         |  Learning Rate: ${metrics.learningRate}%.6f
         |  Processing Time: ${metrics.processingTimeMs}%dms
         |  Memory Usage: ${metrics.memoryUsageMB}%.2fMB""".stripMargin)
  }

  def logEpochMetrics(metrics: EpochMetrics): Unit = {
    logger.info(
      f"""Epoch ${metrics.epochNumber}%d Summary:
         |  Training Loss: ${metrics.loss}%.6f
         |  Training Accuracy: ${metrics.accuracy}%.2f%%
         |  Validation Loss: ${metrics.validationLoss.map(_.formatted("%.6f")).getOrElse("N/A")}
         |  Validation Accuracy: ${metrics.validationAccuracy.map(v => f"$v%.2f%%").getOrElse("N/A")}
         |  Learning Rate: ${metrics.learningRate}%.6f
         |  Duration: ${metrics.durationMs}%dms""".stripMargin)

    logGradientStats(metrics.gradientStats)
  }

  def logFinalMetrics(metrics: TrainingMetrics): Unit = {
    logger.info(
      f"""Training Complete:
         |Total Duration: ${metrics.totalDurationMs}%dms
         |Average Batch Time: ${metrics.averageBatchTimeMs}%.2fms
         |Peak Memory Usage: ${metrics.peakMemoryUsageMB}%.2fMB
         |
         |Model Statistics:
         |  Total Parameters: ${metrics.modelMetrics.totalParameters}%d
         |  Batches Processed: ${metrics.modelMetrics.batchesProcessed}%d
         |  Avg Processing Time/Batch: ${metrics.modelMetrics.averageProcessingTimePerBatchMs}%.2fms
         |
         |Spark Statistics:
         |  Total Executor Memory: ${metrics.sparkMetrics.totalExecutorMemory} MB
         |  Active Executors: ${metrics.sparkMetrics.activeExecutors}%d bytes
         |  Partition Skew: ${metrics.sparkMetrics.dataDistributionStats.partitionSkewPercentage}%.2f%%""".stripMargin)
  }
}

