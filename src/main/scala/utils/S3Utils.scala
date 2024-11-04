package utils

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import scala.io.Source

object S3Utils {
  private val logger = LoggerFactory.getLogger(getClass)

  def readFromS3(sc: SparkContext, path: String): Iterator[String] = {
    val hadoopPath = new Path(path)
    val fs = FileSystem.get(sc.hadoopConfiguration)

    try {
      val inputStream = fs.open(hadoopPath)
      Source.fromInputStream(inputStream).getLines()
    } catch {
      case e: Exception =>
        logger.error(s"Error reading from S3 path $path: ${e.getMessage}")
        throw e
    }
  }

  def writeToS3(sc: SparkContext, content: Array[Byte], path: String): Unit = {
    val hadoopPath = new Path(path)
    val fs = FileSystem.get(sc.hadoopConfiguration)

    try {
      val outputStream = fs.create(hadoopPath)
      outputStream.write(content)
      outputStream.close()
    } catch {
      case e: Exception =>
        logger.error(s"Error writing to S3 path $path: ${e.getMessage}")
        throw e
    }
  }

  def writeStringToS3(sc: SparkContext, content: String, path: String): Unit = {
    writeToS3(sc, content.getBytes("UTF-8"), path)
  }

  def createS3Directory(sc: SparkContext, path: String): Unit = {
    val hadoopPath = new Path(path)
    val fs = FileSystem.get(sc.hadoopConfiguration)

    if (!fs.exists(hadoopPath)) {
      fs.mkdirs(hadoopPath)
    }
  }
}