package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.expressions.base.{CountAccumulatorMap, Utils}
import org.apache.spark.sql.catalyst.expressions.aggregate.{ImperativeAggregate, TypedImperativeAggregate}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.util.MapData
import org.apache.spark.sql.types._

case class ARC_MergeCountMapAgg(
    accumulatorMapExpr: Expression,
    mutableAggBufferOffset: Int = 0,
    inputAggBufferOffset: Int = 0
) extends TypedImperativeAggregate[CountAccumulatorMap] {

    override def children: Seq[Expression] = Seq(accumulatorMapExpr)

    override def createAggregationBuffer(): CountAccumulatorMap = CountAccumulatorMap()

    override def update(buffer: CountAccumulatorMap, input: InternalRow): CountAccumulatorMap = {
        val left = CountAccumulatorMap(accumulatorMapExpr.eval(input).asInstanceOf[MapData])
        merge(buffer, left)
    }

    override def merge(buffer: CountAccumulatorMap, input: CountAccumulatorMap): CountAccumulatorMap = {
        buffer.merge(input)
    }

    override def eval(buffer: CountAccumulatorMap): Any = {
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

    override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression =
        copy(accumulatorMapExpr = newChildren.head)

}
