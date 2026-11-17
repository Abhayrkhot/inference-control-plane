# Tunnel from a pod in Kubernetes to a vLLM job on an HPC compute node, via the login node.
FROM alpine:3.20
RUN apk add --no-cache openssh-client
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
