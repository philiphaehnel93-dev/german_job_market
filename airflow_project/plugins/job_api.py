import os
from flask import Blueprint, send_from_directory
from flask_appbuilder import BaseView, expose

from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Importing base classes that we need to derive
from airflow.hooks.base import BaseHook
#api_fastapi seems to be missing
#from airflow.api_fastapi.core_api.security import GetUserDep, requires_access_dag, get_user

# This is the class you derive to create a plugin
from airflow.plugins_manager import AirflowPlugin


# Creating a FastAPI application to integrate in Airflow Rest API.
app = FastAPI(
    title="Airflow FastAPI Plugin",
    description="REST endpoints powered by FastAPI",
    version="0.1"
    #dependencies=[Depends(get_user)]
)

PLUGIN_DIR = os.path.dirname(os.path.realpath(__file__))
STATIC_DIR = os.path.join(PLUGIN_DIR, "static")

job_api_bp = Blueprint(
    "job_api_bp",
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/job-api/static"
)

#@app.get("/dashboard")
#def dashboard(user: GetUserDep):
#    return {"user": user.get_name()}

@app.get("/")
async def root():
    return {"message": "Hello World from FastAPI plugin"}

@app.get("/health")
async def get_health():
    """
    A custom JSON endpoint that can be queried by frontend frameworks
    or external monitoring tools.
    """
    return {
        "status": "healthy",
        "message": "Hello from Airflow's native FastAPI environment!"
    }

#@app.get("/dags/{dag_id}", dependencies=[Depends(requires_access_dag(method="GET"))])
#def dag_detail(dag_id: str):
#    return {"dag_id": dag_id}

@app.get("/ui", response_class=HTMLResponse)
def get_ui():
    """
    Serves the static index.html dashboard file.
    """
    html_file_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Error: index.html not found in static folder</h1>", status_code=404)


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class JobApiView(BaseView):
    default_view = "index"

    @expose("/")
    def index(self):
        # Serves plugins/static/index.html directly
        return send_from_directory(STATIC_DIR, "index.html")

# Defining the plugin class
class FastApiPlugin(AirflowPlugin):
    name = 'job-api'

    # Serve static files
    fastapi_apps = [
        {
            "app": app,
            "url_prefix": "/job-api",
            "name": "German Jobs API End Point powered by FastAPI",
        }
    ]

    flask_blueprints = [job_api_bp]

    appbuilder_menu_items = [
        {
            "name": "Custom API Overview",
            "category": "Admin",
            "category_icon": "fa-cog",
            "href": "/job-api/ui",
        }
    ]

    appbuilder_views = [
        {
            "name": "Job API",
            "category": "Custom Views",
            "view": JobApiView(),
        }
    ]
