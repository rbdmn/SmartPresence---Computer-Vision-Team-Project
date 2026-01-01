from fastapi import APIRouter, HTTPException
from bson import ObjectId
from config.configrations import class_collection

router = APIRouter()


def normalize(doc: dict):
	if not doc:
		return doc
	d = dict(doc)
	# convert ObjectId to string for _id
	_id = d.get("_id")
	if _id is not None:
		try:
			d["_id"] = str(_id)
		except Exception:
			d["_id"] = _id
	return d


@router.get("/", tags=["Class"])
async def list_classes():
	items = []
	for c in class_collection.find():
		items.append(normalize(c))
	return items


@router.get("/{id}", tags=["Class"])
async def get_class(id: str):
	try:
		obj_id = ObjectId(id)
	except Exception:
		raise HTTPException(status_code=400, detail="Invalid ObjectId")
	c = class_collection.find_one({"_id": obj_id})
	if not c:
		raise HTTPException(status_code=404, detail="Class not found")
	return normalize(c)

