package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.expressions.base.{CountAccumulatorMap, Utils}
import org.apache.spark.sql.catalyst.expressions.aggregate.{ImperativeAggregate, TypedImperativeAggregate}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.types._

case class ARC_CombinatorialCountAgg(
    attributeExprs: Seq[Expression],
    attributeNames: Seq[String],
    nCombination: Int = 2,
    mutableAggBufferOffset: Int = 0,
    inputAggBufferOffset: Int = 0,
    threshold: Int = 1
) extends TypedImperativeAggregate[CountAccumulatorMap] {

    private val attributeMap = attributeNames.zip(attributeExprs).toMap
    private val reverseMap = attributeMap.map(_.swap)
    private val combinations = attributeMap.values.toList.combinations(nCombination).toList

    override def children: Seq[Expression] = attributeMap.values.toSeq

    override def createAggregationBuffer(): CountAccumulatorMap = CountAccumulatorMap()

    override def update(buffer: CountAccumulatorMap, input: InternalRow): CountAccumulatorMap = {
        val left = combinations.map(
          combination => {
              val combKey = combination.map(reverseMap).mkString("", ",", ";")
              val combValue = combination.map(_.eval(input).toString).mkString("", ",", "")
              combKey + combValue
          }
        )
        buffer.merge(CountAccumulatorMap(left))
    }

    override def merge(buffer: CountAccumulatorMap, input: CountAccumulatorMap): CountAccumulatorMap = {
        val pre = buffer.merge(input).counter
        val valid_count = pre.filter(_._2 > threshold)
        CountAccumulatorMap(valid_count)
    }

    override def eval(buffer: CountAccumulatorMap): Any = {
        val counter = buffer.counter.toList.sortBy(c => -c._2).take(1000).toMap
        val result = Utils.buildMapLong(buffer.counter)
        result
    }

    override def serialize(buffer: CountAccumulatorMap): Array[Byte] = {
        Utils.serialize(buffer)
    }

    override def deserialize(storageFormat: Array[Byte]): CountAccumulatorMap = {
        Utils.deserialize[CountAccumulatorMap](storageFormat)
    }

    override def withNewMutableAggBufferOffset(newMutableAggBufferOffset: Int): ImperativeAggregate =
        copy(mutableAggBufferOffset = newMutableAggBufferOffset)

    override def withNewInputAggBufferOffset(newInputAggBufferOffset: Int): ImperativeAggregate =
        copy(inputAggBufferOffset = newInputAggBufferOffset)

    override def nullable: Boolean = false

    override def dataType: DataType = MapType(StringType, LongType)

    override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression = copy(attributeExprs = newChildren)

}
