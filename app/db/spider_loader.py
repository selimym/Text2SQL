import json

from app.db.schema_doc import ColumnInfo, ForeignKeyInfo, SchemaDocument


def load_schema_documents_from_spider(tables_json_path: str) -> list[SchemaDocument]:
    """Load SchemaDocuments from a Spider dataset tables.json file."""
    with open(tables_json_path) as f:
        databases = json.load(f)

    all_docs: list[SchemaDocument] = []
    for db in databases:
        db_id: str = db["db_id"]
        table_names: list[str] = db["table_names_original"]
        raw_columns: list[list[int | str]] = db["column_names_original"]
        col_types: list[str] = db["column_types"]
        primary_keys: list[int] = db["primary_keys"]
        foreign_keys: list[list[int]] = db["foreign_keys"]

        # Build column map (index → info), skipping the wildcard at index 0
        # raw_columns[i] = [table_idx, col_name]; table_idx == -1 means wildcard
        col_map: dict[int, tuple[int, str]] = {}  # idx → (table_idx, col_name)
        for i, (table_idx, col_name) in enumerate(raw_columns):
            if table_idx == -1:
                continue
            col_map[i] = (int(table_idx), str(col_name))

        pk_set = set(primary_keys)

        # Group columns by table
        table_cols: dict[int, list[ColumnInfo]] = {i: [] for i in range(len(table_names))}
        for col_idx, (table_idx, col_name) in col_map.items():
            # col_types index matches raw_columns index
            col_type = col_types[col_idx] if col_idx < len(col_types) else "text"
            table_cols[table_idx].append(
                ColumnInfo(
                    name=col_name,
                    data_type=col_type,
                    is_primary_key=col_idx in pk_set,
                )
            )

        # Build FK map: from_table_idx → list of ForeignKeyInfo
        table_fks: dict[int, list[ForeignKeyInfo]] = {i: [] for i in range(len(table_names))}
        for from_idx, to_idx in foreign_keys:
            if from_idx not in col_map or to_idx not in col_map:
                continue
            from_table_idx, from_col_name = col_map[from_idx]
            to_table_idx, to_col_name = col_map[to_idx]
            table_fks[from_table_idx].append(
                ForeignKeyInfo(
                    from_column=from_col_name,
                    to_table=table_names[to_table_idx],
                    to_column=to_col_name,
                )
            )

        for table_idx, table_name in enumerate(table_names):
            all_docs.append(
                SchemaDocument(
                    db_id=db_id,
                    table_name=table_name,
                    columns=table_cols[table_idx],
                    foreign_keys=table_fks[table_idx],
                )
            )

    return all_docs
