FROM ubuntu:24.04
RUN apt-get update && apt-get install -y sshpass openssh-client netcat-openbsd coreutils && rm -rf /var/lib/apt/lists/*
COPY benign.sh /benign.sh
RUN chmod +x /benign.sh
CMD ["/benign.sh"]
