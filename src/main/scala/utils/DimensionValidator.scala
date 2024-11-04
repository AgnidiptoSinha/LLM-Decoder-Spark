package utils

import org.slf4j.LoggerFactory

case class DimensionConfig(
                            windowSize: Int,
                            embeddingSize: Int,
                            inputSize: Int,
                            outputSize: Int,
                            hiddenSize: Int
                          )

object DimensionValidator {
  private val logger = LoggerFactory.getLogger(getClass)

  def validateDimensions(config: DimensionConfig): Boolean = {
    try {
      val expectedInputSize = config.windowSize * config.embeddingSize

      require(config.windowSize > 0,
        s"Window size must be positive, got ${config.windowSize}")

      require(config.embeddingSize > 0,
        s"Embedding size must be positive, got ${config.embeddingSize}")

      require(config.inputSize == expectedInputSize,
        s"Input size mismatch: expected $expectedInputSize (windowSize * embeddingSize) but got ${config.inputSize}")

      require(config.outputSize == config.embeddingSize,
        s"Output size mismatch: expected ${config.embeddingSize} (embedding size) but got ${config.outputSize}")

      require(config.hiddenSize >= config.embeddingSize,
        s"Hidden size must be at least as large as embedding size, got ${config.hiddenSize} < ${config.embeddingSize}")

      logger.info("Dimension validation successful:")
      logger.info(s"Window size: ${config.windowSize}")
      logger.info(s"Embedding size: ${config.embeddingSize}")
      logger.info(s"Input size: ${config.inputSize}")
      logger.info(s"Output size: ${config.outputSize}")
      logger.info(s"Hidden size: ${config.hiddenSize}")

      true
    } catch {
      case e: IllegalArgumentException =>
        logger.error(s"Dimension validation failed: ${e.getMessage}")
        throw e
    }
  }
}