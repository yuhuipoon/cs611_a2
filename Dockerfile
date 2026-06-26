# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1-python3.9

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME directory and symlink the java binary there
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Bake the static input CSVs into the image at /data. This avoids bind-mounting
# ./data through the macOS Docker Desktop FUSE layer, which intermittently fails
# Spark reads with "Resource deadlock avoided" (EDEADLK). /data lives on the
# image overlay FS, so reads never touch FUSE.
COPY --chown=airflow:0 data/ /data/

# Bake dags and utils into the image — bind-mounting these through the macOS
# Docker Desktop FUSE layer triggers EDEADLK when the scheduler reads DAG files.
COPY --chown=airflow:0 dags/ /dags/
COPY --chown=airflow:0 utils/ /utils/
RUN ln -s /utils /opt/airflow/utils

# Bake the pre-built datamart into the image at the volume mount path. A fresh
# Docker named volume auto-populates from whatever the image has at its mount
# point, so on the first `docker compose up` the volume comes up pre-seeded with
# the full bronze→silver→gold history (markers included) — no manual sync needed,
# reproducible from a clean checkout. (COPY preserves the hidden .done_* markers,
# unlike `docker cp dir/.` which silently drops dotfiles.)
#
# datamart/ is a Docker named volume (off the macOS FUSE bind mount to avoid the
# EDEADLK that hits Spark JVM reads). The volume inherits the ownership of the
# image dir it mounts over, hence --chown so the airflow user can write.
# NOTE: model_registry/ and datamart/gold/inference/ are HOST bind mounts (see
# docker-compose.yaml) so outputs stay live on the host; the bind mount shadows
# whatever is baked here for those paths.
COPY --chown=airflow:0 datamart/ /opt/airflow/datamart/
RUN mkdir -p /opt/airflow/model_registry && \
    chown -R airflow:0 /opt/airflow/datamart /opt/airflow/model_registry

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app
