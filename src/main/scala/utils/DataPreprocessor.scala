package utils

import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.{INDArrayIndex, NDArrayIndex}
import org.slf4j.LoggerFactory

case class DataPreprocessor(inputSize: Int, outputSize: Int) extends Serializable {
  @transient private lazy val logger = LoggerFactory.getLogger(getClass)

  def preprocess(dataset: DataSet): DataSet = {
    // Ensure input features are shaped correctly
    val features = dataset.getFeatures
    val labels = dataset.getLabels

    // Log the shapes for debugging
    logger.debug(s"Original features shape: ${features.shape().mkString("x")}")
    logger.debug(s"Original labels shape: ${labels.shape().mkString("x")}")

    // Calculate the actual input size from the window size and embedding dimension
    val actualInputSize = features.shape()(1)

    // Only reshape if dimensions don't match expected sizes
    val reshapedFeatures = if (features.shape()(1) != inputSize) {
      if (actualInputSize > inputSize) {
        // If input is larger than expected, truncate
        features.get(NDArrayIndex.all(), NDArrayIndex.interval(0, inputSize))
      } else if (actualInputSize < inputSize) {
        // If input is smaller than expected, pad with zeros
        val padded = Nd4j.zeros(features.shape()(0), inputSize)
        padded.put(
          Array[INDArrayIndex](
            NDArrayIndex.all(),
            NDArrayIndex.interval(0, actualInputSize)
          ),
          features
        )
        padded
      } else {
        features
      }
    } else {
      features
    }

    // Similar process for labels
    val actualOutputSize = labels.shape()(1)
    val reshapedLabels = if (labels.shape()(1) != outputSize) {
      if (actualOutputSize > outputSize) {
        labels.get(NDArrayIndex.all(), NDArrayIndex.interval(0, outputSize))
      } else if (actualOutputSize < outputSize) {
        val padded = Nd4j.zeros(labels.shape()(0), outputSize)
        padded.put(
          Array[INDArrayIndex](
            NDArrayIndex.all(),
            NDArrayIndex.interval(0, actualOutputSize)
          ),
          labels
        )
        padded
      } else {
        labels
      }
    } else {
      labels
    }

    logger.debug(s"Reshaped features shape: ${reshapedFeatures.shape().mkString("x")}")
    logger.debug(s"Reshaped labels shape: ${reshapedLabels.shape().mkString("x")}")

    new DataSet(reshapedFeatures, reshapedLabels)
  }
}