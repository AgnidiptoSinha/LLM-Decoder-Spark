ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.18"

val sparkVersion = "3.5.1"
val dl4jVersion = "1.0.0-beta7"
val nd4jVersion = "1.0.0-beta7"

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", "LICENSE") => MergeStrategy.discard
  case PathList("META-INF", "License") => MergeStrategy.discard
  case PathList("META-INF", "LICENSE.txt") => MergeStrategy.discard
  case PathList("META-INF", "License.txt") => MergeStrategy.discard
  case PathList("META-INF", "license") => MergeStrategy.discard
  case PathList("META-INF", "license.txt") => MergeStrategy.discard
  case PathList("META-INF", xs @ _*) =>
    xs match {
      case "MANIFEST.MF" :: Nil => MergeStrategy.discard
      case "services" :: _ => MergeStrategy.concat
      case _ => MergeStrategy.discard
    }
  case "reference.conf" => MergeStrategy.concat
  case x if x.endsWith(".proto") => MergeStrategy.rename
  case x if x.contains("hadoop") => MergeStrategy.first
  case x if x.endsWith(".properties") => MergeStrategy.first
  case PathList("org", "bytedeco", xs @ _*) => MergeStrategy.first
  case PathList("org", "nd4j", xs @ _*) => MergeStrategy.first
  case PathList("native", xs @ _*) => MergeStrategy.first
  case _ => MergeStrategy.first
}

fork := true

lazy val root = (project in file("."))
  .settings(
    name := "LLM-Decoder-Spark",
    assembly / mainClass := Some("hw2.main"),

    libraryDependencies ++= Seq(
      // Spark dependencies - no longer "provided"
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,

      // DL4J & ND4J dependencies
      "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
      "org.nd4j" % "nd4j-native" % nd4jVersion classifier "" classifier "linux-x86_64",
      "org.nd4j" % "nd4j-common" % nd4jVersion,
      "org.nd4j" % "nd4j-api" % nd4jVersion,
      "org.deeplearning4j" % "deeplearning4j-core" % dl4jVersion,
      "org.deeplearning4j" % "dl4j-spark_2.12" % dl4jVersion,

      // Logging & Config
      "ch.qos.logback" % "logback-classic" % "1.2.11",
      "com.typesafe" % "config" % "1.4.2",

      // Test Dependencies
      "org.scalatest" %% "scalatest" % "3.2.15" % Test,
      "org.apache.spark" %% "spark-core" % sparkVersion % Test,
      "org.apache.spark" %% "spark-sql" % sparkVersion % Test

    ),

  )

resolvers += "Maven Central" at "https://repo1.maven.org/maven2/"
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"

// Add explicit main class setting
Compile / run / mainClass := Some("hw2.main")