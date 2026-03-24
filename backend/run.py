from routes import app
import routes
from flask_cors import CORS
import os
from flask import Flask,send_from_directory    
    
CORS(app)

# if __name__ == '__main__':
#     port = int(os.getenv('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == "__main__":
    app.run()