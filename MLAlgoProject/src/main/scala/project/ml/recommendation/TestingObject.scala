package project.ml.recommendation

import java.net.URL
import java.io.File
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame,SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext


object TestingObject {
  def main(args: Array[String]) {
    
    val tmpFile = new File("C:\\Users\\Rahul\\Desktop\\rws.json")
    FileUtils.copyURLToFile(new URL("https://health.data.ny.gov/api/views/jxy9-yhdk/rows.json?accessType=DOWNLOAD"), tmpFile)
    
    val spark =SparkSession.builder().appName("test").master("local[*]").getOrCreate()
    spark.read.json("C:\\Users\\Rahul\\Desktop\\rws.json").printSchema()
    
    
    
  }
}