package utils

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator

import scala.collection.mutable.ListBuffer
import collection.JavaConverters._

class SlidingWindowDataset(
                            tokens: Array[String],
                            embeddings: Map[String, INDArray],
                            windowSize: Int,
                            embeddingSize: Int
                          ) {

  def createSlidingWindows(): List[DataSet] = {
    val windowedData = new ListBuffer[DataSet]()

    for (i <- 0 until tokens.length - windowSize) {
      val inputWindow = tokens.slice(i, i + windowSize)
      println("Input Window :"+ inputWindow.mkString(", "))
      val targetToken = tokens(i + windowSize)
      println("Target Token :"+ targetToken)

      val inputEmbeddings = combineEmbeddings(inputWindow)
      val targetEmbedding = embeddings(targetToken)

      val input = inputEmbeddings.reshape(1, embeddingSize * windowSize)
      val target = targetEmbedding.reshape(1, embeddingSize)

      windowedData += new DataSet(input, target)
    }

    windowedData.toList
  }

  private def combineEmbeddings(window: Array[String]): INDArray = {
    val combinedEmbedding = Nd4j.zeros(window.length, embeddingSize)

    for (i <- window.indices) {
      val embedding = embeddings(window(i))
      println("Embeddings :"+embedding)
      val positionalEmbedding = computePositionalEmbedding(i)
      println(s"Positional Embeddings at ${i}:"+positionalEmbedding.getRow(0))
      combinedEmbedding.putRow(i, embedding.add(positionalEmbedding))

      }

//    println("Combined embeddings :"+combinedEmbedding.getRow(0))

    combinedEmbedding
  }

  private def computePositionalEmbedding(position: Int): INDArray = {
    val posEmbedding = Nd4j.zeros(embeddingSize)

    for (i <- 0 until embeddingSize by 2) {
      val angle = position / Math.pow(10000, (2.0 * i) / embeddingSize)
      posEmbedding.putScalar(i, Math.sin(angle))
      posEmbedding.putScalar(i + 1, Math.cos(angle))
    }

    posEmbedding
  }
}

object SlidingWindowExample {
  def main(args: Array[String]): Unit = {
    // Sample data (replace with your actual embeddings from homework 1)
    val sampleTokens = Array("The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog")
    val embeddingSize = 128
    val sampleEmbeddings = sampleTokens.map(token => (token, Nd4j.rand(embeddingSize))).toMap

    println(sampleEmbeddings)

    val windowSize = 4
    val slidingWindowDataset = new SlidingWindowDataset(sampleTokens, sampleEmbeddings, windowSize, embeddingSize)
    val datasetList = slidingWindowDataset.createSlidingWindows()

    println(s"Created ${datasetList.size} sliding windows")

    // Create a DataSetIterator for training
    val batchSize = 2
    val iterator = new ListDataSetIterator[DataSet](datasetList.asJava, batchSize)

    // Print some information about the first batch
    val firstBatch = iterator.next()
    println(s"Input shape: ${firstBatch.getFeatures.shape().mkString("x")}")
    println(s"Target shape: ${firstBatch.getLabels.shape().mkString("x")}")
  }
}