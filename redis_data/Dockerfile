# Use the official Redis image as the base image
FROM redis
# Expose the default Redis port, so the service can be accessed
EXPOSE 6379
# Create a volume
VOLUME /data
COPY redis_entrypoint.sh redis_entrypoint.sh
RUN chmod +x redis_entrypoint.sh
# Use an environment variable for the password
# Start Redis server when the container runs
# The server will automatically load the dump.rdb file from the /data directory
CMD ["./redis_entrypoint.sh"]