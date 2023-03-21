package com.databricks.industry.solutions.arc.expressions

import com.databricks.industry.solutions.arc.expressions.ARC_Combinations.asStringSeq
import org.apache.spark.sql.catalyst.expressions.{CollectionGenerator, Expression}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.CodegenFallback
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

case class ARC_GeneratePartialCombinations(
    nCombinationsExpr: Expression,
    partialsExpr: Expression,
    elementsExpr: Expression
) extends CollectionGenerator
      with Serializable
      with CodegenFallback {

    override def position: Boolean = false

    override def inline: Boolean = false

    override def children: Seq[Expression] = Seq(nCombinationsExpr, partialsExpr, elementsExpr)

    override def eval(input: InternalRow): TraversableOnce[InternalRow] = {
        val arrayDataElements = elementsExpr.eval(input).asInstanceOf[ArrayData]
        val arrayDataPartials = partialsExpr.eval(input).asInstanceOf[ArrayData]
        val nComb = nCombinationsExpr.eval(input).asInstanceOf[Int]
        val elements = asStringSeq(arrayDataElements)
        val partials = asStringSeq(arrayDataPartials)

        val combinations = elements
            .diff(partials)
            .combinations(nComb)
            .toArray
            .map(combination => partials ++ combination)

        val result = if (combinations.isEmpty | nComb < 1) Array(partials) else combinations

        result
            .map(_.sorted)
            .map(combination => ArrayData.toArrayData(combination.map(UTF8String.fromString)))
            .map(row => InternalRow.fromSeq(Seq(row)))
    }

    override def elementSchema: StructType = StructType(Seq(StructField("combination", ArrayType(StringType))))

    override def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): Expression =
        copy(newChildren(0), newChildren(1), newChildren(2))

}
