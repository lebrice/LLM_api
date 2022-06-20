# SBATCH

#  TODO: Write an sbatch script to start the server with a given model capacity.
export DEFAULT_CAPACITY="13b"
export SERVER_PORT=12345

uvicorn server:app --port $SERVER_PORT
