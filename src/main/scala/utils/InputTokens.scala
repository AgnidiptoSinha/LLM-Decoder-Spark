package utils

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.SparkContext
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.util.{Failure, Success, Try}

case class TokenEmbedding(token: Int, embedding: INDArray) extends Serializable {
  // Add a custom serialization method if needed
  @transient private lazy val logger = LoggerFactory.getLogger(getClass)
}

class InputTokens(sc: SparkContext) {
  private val logger = LoggerFactory.getLogger(getClass)
  private val config = ConfigFactory.load()

  def loadTokenEmbeddings(filePath: String): Try[Vector[TokenEmbedding]] = {
    Try {
      val lines = S3Utils.readFromS3(sc, filePath)
      lines.map(parseLine).toVector
    }
  }

  private def parseLine(line: String): TokenEmbedding = {
    val parts = line.split("\t")
    require(parts.length == 2, s"Invalid line format: $line")

    val token = parts(0).toInt
    val embeddingString = parts(1).stripPrefix("[").stripSuffix("]")
    val embeddingValues = embeddingString.split(",").map(_.toDouble)
    val embedding = Nd4j.create(embeddingValues)

    TokenEmbedding(token, embedding)
  }

  def getEmbeddingSize(embeddings: Vector[TokenEmbedding]): Int = {
    if (embeddings.isEmpty) 0
    else embeddings.head.embedding.length().toInt
  }

  def validateEmbeddings(embeddings: Vector[TokenEmbedding]): Boolean = {
    if (embeddings.isEmpty) {
      logger.error("No embeddings found")
      return false
    }

    val embeddingSize = getEmbeddingSize(embeddings)
    val allSameSize = embeddings.forall(_.embedding.length() == embeddingSize)

    if (!allSameSize) {
      logger.error("Inconsistent embedding sizes found")
      return false
    }

    true
  }
}

object InputTokens {
  def apply(sc: SparkContext): InputTokens = new InputTokens(sc)
}