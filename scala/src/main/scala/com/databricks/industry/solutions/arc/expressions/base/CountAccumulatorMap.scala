package com.databricks.industry.solutions.arc.expressions.base

import org.apache.spark.sql.catalyst.util.MapData
import org.apache.spark.sql.types.{LongType, StringType}
import org.apache.spark.unsafe.types.UTF8String

case class CountAccumulatorMap(counter: Map[String, Long]) extends Serializable {

    def this() = this(Map.empty)

    def merge(other: CountAccumulatorMap): CountAccumulatorMap = {
        val newKeys = other.counter.keySet ++ counter.keySet
        val newMap_pre = newKeys.map { k => k -> (other.counter.getOrElse(k, 0L) + counter.getOrElse(k, 0L)) }.toMap
        val newMap = newMap_pre.filter(_._2 > 1)
        CountAccumulatorMap(newMap)
    }

    def ++(key: String): CountAccumulatorMap = {
        val newMap = counter + (key -> (counter.getOrElse(key, 0L) + 1L))
        CountAccumulatorMap(newMap)
    }

}

object CountAccumulatorMap {

    def apply(): CountAccumulatorMap = CountAccumulatorMap(Map.empty[String, Long])

    def apply(keys: Seq[String]): CountAccumulatorMap = CountAccumulatorMap(keys.map(k => k -> 1L).toMap)

    def apply(mapData: MapData): CountAccumulatorMap = {
        val keys = mapData.keyArray().toSeq[UTF8String](StringType).map(_.toString)
        val values = mapData.   valueArray().toSeq[Long](LongType)
        new CountAccumulatorMap(keys.zip(values).toMap)
    }

}
