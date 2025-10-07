from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
from datetime import datetime

router = APIRouter()

@router.post("/")
async def export_results(request: Request):
    data = await request.json()
    if not data:
        return {"error": "Aucune donnée reçue."}

    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")

    # Formatage du fichier Excel
    with pd.ExcelWriter(temp_file.name, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Résultats")

        ws = writer.sheets["Résultats"]
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
            cell.fill = cell.fill.copy(start_color="DDDDDD", end_color="DDDDDD", fill_type="solid")

        for col in ws.columns:
            max_length = max(len(str(cell.value)) for cell in col if cell.value)
            ws.column_dimensions[col[0].column_letter].width = max_length + 2

    filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return FileResponse(temp_file.name, filename=filename)
