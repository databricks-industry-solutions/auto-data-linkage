package com.databricks.industry.solutions.arc.expressions

import org.apache.spark.sql.catalyst.expressions.{BinaryExpression, Expression}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

case class ARC_Combinations(
    nCombinationsExpr: Expression,
    elementsExpr: Expression
) extends BinaryExpression
      with Serializable
      with CodegenFallback {

    override def eval(input: InternalRow): Any = {
        val combinations = ARC_Combinations.evalCombinations(input, nCombinationsExpr, elementsExpr)
        ArrayData.toArrayData(combinations)
    }

    override def left: Expression = nCombinationsExpr

    override def right: Expression = elementsExpr

    override def dataType: DataType = ArrayType(ArrayType(StringType))

    override protected def withNewChildrenInternal(newLeft: Expression, newRight: Expression): Expression =
        ARC_Combinations(newLeft, newRight)

}

object ARC_Combinations {

    def evalCombinations(input: InternalRow, nCombinationsExpr: Expression, elementsExpr: Expression): Array[ArrayData] = {
        val arrayData = elementsExpr.eval(input).asInstanceOf[ArrayData]
        val nComb = nCombinationsExpr.eval(input).asInstanceOf[Int]
        val elements = asStringSeq(arrayData)
        elements
            .combinations(nComb)
            .toArray
            .map(combination => ArrayData.toArrayData(combination.map(UTF8String.fromString)))
    }

    def asStringSeq(arrayData: ArrayData): Seq[String] = {
        arrayData
            .toArray[UTF8String](StringType)
            .toSeq
            .asInstanceOf[Seq[UTF8String]]
            .map(_.toString)
    }

}
