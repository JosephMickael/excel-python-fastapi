from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import ORJSONResponse
from services.douteux_utils import find_doubtful_matches_main

router = APIRouter()

@router.post("/", response_class=ORJSONResponse)
async def detect_douteux(
    file_ifs: UploadFile = File(...),
    file_topview: UploadFile = File(...)
):
    """
    Endpoint indépendant pour détecter les correspondances douteuses entre IFS et TopView
    """
    try:
        result = find_doubtful_matches_main(file_ifs, file_topview)
        return ORJSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
