from app.api.models import ExecutionMetadata, QueryRequest, QueryResponse
from app.db.executor import SQLExecutor
from app.db.validator import SQLValidator
from app.llm.generator import SQLGenerator
from app.pipeline.assembler import PromptAssembler
from app.retrieval.example_retriever import ExampleRetriever
from app.retrieval.schema_retriever import SchemaRetriever


class BaselinePipeline:
    def __init__(
        self,
        schema_retriever: SchemaRetriever,
        example_retriever: ExampleRetriever,
        assembler: PromptAssembler,
        generator: SQLGenerator,
        validator: SQLValidator,
        executor: SQLExecutor,
        spider_data_dir: str,
        schema_similarity_threshold: float | None = None,
    ) -> None:
        self.schema_retriever = schema_retriever
        self.example_retriever = example_retriever
        self.assembler = assembler
        self.generator = generator
        self.validator = validator
        self.executor = executor
        self.spider_data_dir = spider_data_dir
        self.schema_similarity_threshold = schema_similarity_threshold

    async def run(self, request: QueryRequest) -> QueryResponse:
        if self.schema_similarity_threshold is not None:
            schema_docs = await self.schema_retriever.retrieve(
                request.question,
                request.db_id,
                request.top_k_schema,
                similarity_threshold=self.schema_similarity_threshold,
            )
        else:
            schema_docs = await self.schema_retriever.retrieve(
                request.question, request.db_id, request.top_k_schema
            )
        example_docs = await self.example_retriever.retrieve(
            request.question, db_id=request.db_id, top_k=request.top_k_examples
        )
        prompt = self.assembler.assemble(request.question, schema_docs, example_docs)
        sql, usage = await self.generator.agenerate(prompt)
        validation = self.validator.validate(sql)
        if not validation.valid:
            return QueryResponse(
                question=request.question,
                generated_sql=sql,
                answer="",
                flags=["validation_failed", validation.error or ""],
                step_timings={"usage": usage} if usage else None,
            )
        db_path = f"{self.spider_data_dir}/database/{request.db_id}/{request.db_id}.sqlite"
        exec_result = await self.executor.execute(sql, db_path)
        answer = (
            str(exec_result.rows)
            if exec_result.success
            else f"Execution failed: {exec_result.error}"
        )
        return QueryResponse(
            question=request.question,
            generated_sql=sql,
            answer=answer,
            retrieved_schema_summary=[d.table_name for d in schema_docs],
            retrieved_examples_summary=[e.question for e in example_docs],
            execution_metadata=ExecutionMetadata(
                success=exec_result.success,
                row_count=exec_result.row_count,
                latency_ms=exec_result.latency_ms,
                error=exec_result.error,
            ),
            step_timings={"usage": usage} if usage else None,
        )
