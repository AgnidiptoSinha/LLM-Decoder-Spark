package metrics

import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory
import utils.S3Utils

import java.time.format.DateTimeFormatter
import java.time.{Instant, ZoneId}
import scala.util.Try

object MetricsOutputHandler {
  private val logger = LoggerFactory.getLogger(getClass)
  private val dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")
    .withZone(ZoneId.systemDefault())

  def saveMetricsToFile(
                         metrics: TrainingMetrics,
                         outputDir: String,
                         sc: SparkContext
                       ): Try[String] = {
    Try {
      // Generate timestamp and full S3 path
      val timestamp = dateFormatter.format(Instant.now())
      val metricsPath = s"$outputDir/training_metrics_$timestamp.txt"

      // Create metrics output
      val metricsOutput = MetricsOutput(
        timestamp = timestamp,
        trainingDuration = formatDuration(metrics.totalDurationMs),
        totalEpochs = metrics.epochs.size,
        finalTrainingAccuracy = metrics.epochs.last.accuracy,
        finalValidationAccuracy = metrics.epochs.last.validationAccuracy,
        finalLoss = metrics.epochs.last.loss,
        finalValidationLoss = metrics.epochs.last.validationLoss,
        averageBatchTime = metrics.averageBatchTimeMs,
        peakMemoryUsage = metrics.peakMemoryUsageMB,
        modelStats = metrics.modelMetrics,
        sparkStats = metrics.sparkMetrics,
        epochMetrics = metrics.epochs
      )

      // Format metrics content
      val content = new StringBuilder()

      content.append("=== LLM Training Metrics Report ===\n")
      content.append(s"Generated: ${metricsOutput.timestamp}\n")

      content.append("\n=== Training Summary ===\n")
      content.append(s"Total Duration: ${metricsOutput.trainingDuration}\n")
      content.append(s"Total Epochs: ${metricsOutput.totalEpochs}\n")
      content.append(f"Final Training Accuracy: ${metricsOutput.finalTrainingAccuracy}%.2f%%\n")
      metricsOutput.finalValidationAccuracy.foreach(acc =>
        content.append(f"Final Validation Accuracy: $acc%.2f%%\n"))
      content.append(f"Final Loss: ${metricsOutput.finalLoss}%.6f\n")
      metricsOutput.finalValidationLoss.foreach(loss =>
        content.append(f"Final Validation Loss: $loss%.6f\n"))
      content.append(f"Average Batch Time: ${metricsOutput.averageBatchTime}%.2f ms\n")
      content.append(f"Peak Memory Usage: ${metricsOutput.peakMemoryUsage}%.2f MB\n")

      content.append("\n=== Model Statistics ===\n")
      content.append(s"Total Parameters: ${metricsOutput.modelStats.totalParameters}\n")
      content.append(s"Layer Sizes: ${metricsOutput.modelStats.layerSizes.mkString(" → ")}\n")
      content.append(s"Total Batches Processed: ${metricsOutput.modelStats.batchesProcessed}\n")
      content.append(f"Average Processing Time per Batch: ${metricsOutput.modelStats.averageProcessingTimePerBatchMs}%.2f ms\n")

      content.append("\n=== Spark Execution Statistics ===\n")
      content.append(s"Active Executors: ${metricsOutput.sparkStats.activeExecutors}\n")
      content.append(f"Total Executor Memory: ${metricsOutput.sparkStats.totalExecutorMemory}%.2f MB\n")
      content.append(s"Active Jobs: ${metricsOutput.sparkStats.activeJobCount}\n")
      content.append(f"Data Partition Skew: ${metricsOutput.sparkStats.dataDistributionStats.partitionSkewPercentage}%.2f%%\n")

      content.append("\n=== Executor Details ===\n")
      metricsOutput.sparkStats.executorMetrics.foreach { executor =>
        content.append(s"Executor (${executor.host}:${executor.port}):\n")
        content.append(f"  Storage Memory Used/Total: ${executor.usedStorageMemoryMB}%.2f MB / ${executor.totalStorageMemoryMB}%.2f MB\n")
        content.append(s"  Running Tasks: ${executor.numRunningTasks}\n")
      }

      content.append("\n=== Epoch Details ===\n")
      metricsOutput.epochMetrics.foreach { epoch =>
        content.append(s"\nEpoch ${epoch.epochNumber}:\n")
        content.append(f"  Loss: ${epoch.loss}%.6f\n")
        content.append(f"  Accuracy: ${epoch.accuracy}%.2f%%\n")
        epoch.validationLoss.foreach(loss =>
          content.append(f"  Validation Loss: $loss%.6f\n"))
        epoch.validationAccuracy.foreach(acc =>
          content.append(f"  Validation Accuracy: $acc%.2f%%\n"))
        content.append(f"  Learning Rate: ${epoch.learningRate}%.6e\n")
        content.append(s"  Duration: ${formatDuration(epoch.durationMs)}\n")

        content.append("  Gradient Statistics:\n")
        content.append(f"    Norm: ${epoch.gradientStats.norm}%.6f\n")
        content.append(f"    Min/Max: ${epoch.gradientStats.min}%.6f / ${epoch.gradientStats.max}%.6f\n")
        content.append(f"    Mean: ${epoch.gradientStats.mean}%.6f\n")
        content.append(f"    Std Dev: ${epoch.gradientStats.standardDeviation}%.6f\n")
      }

      // Create the output directory if it doesn't exist
      S3Utils.createS3Directory(sc, outputDir)

      // Write the formatted content to S3
      S3Utils.writeStringToS3(sc, content.toString(), metricsPath)
      logger.info(s"Training metrics saved to: $metricsPath")

      metricsPath
    }
  }

  private def formatDuration(ms: Long): String = {
    val seconds = ms / 1000
    val minutes = seconds / 60
    val hours = minutes / 60

    if (hours > 0) {
      f"${hours}h ${minutes % 60}m ${seconds % 60}s"
    } else if (minutes > 0) {
      f"${minutes}m ${seconds % 60}s"
    } else {
      f"${seconds}s"
    }
  }
}

case class MetricsOutput(
                          timestamp: String,
                          trainingDuration: String,
                          totalEpochs: Int,
                          finalTrainingAccuracy: Double,
                          finalValidationAccuracy: Option[Double],
                          finalLoss: Double,
                          finalValidationLoss: Option[Double],
                          averageBatchTime: Double,
                          peakMemoryUsage: Double,
                          modelStats: ModelMetrics,
                          sparkStats: SparkTrainingMetrics,
                          epochMetrics: Vector[EpochMetrics]
                        )