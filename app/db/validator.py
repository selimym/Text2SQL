import sqlglot
from pydantic import BaseModel


class ValidationResult(BaseModel):
    valid: bool
    error: str | None = None


class SQLValidator:
    def validate(self, sql: str) -> ValidationResult:
        sql = sql.strip()
        if not sql:
            return ValidationResult(valid=False, error="Empty SQL")
        try:
            statements = sqlglot.parse(sql)
        except sqlglot.errors.ParseError as e:
            return ValidationResult(valid=False, error=f"Parse error: {e}")
        if len(statements) != 1:
            return ValidationResult(valid=False, error="Only single statements allowed")
        stmt = statements[0]
        if not isinstance(stmt, sqlglot.exp.Select):
            return ValidationResult(
                valid=False,
                error=f"Only SELECT allowed, got: {type(stmt).__name__}",
            )
        return ValidationResult(valid=True)
