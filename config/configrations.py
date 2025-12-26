from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://raufbdmn_db_user:ShKGiD8Lcj9mzB1o@smartpresencecluster.or44jha.mongodb.net/?appName=SmartPresenceCluster"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["ComputerVisionProject"]
users_collection = db["Users"]
attendance_collection = db["Attendance"]
vector_collection = db["Vector"]
visitor_vector_collection = db["VisitorVector"]
visitor_collection = db["Visitors"]