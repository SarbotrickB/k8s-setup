# k8s-setup
Technical Assessment to set up a Kubernetes cluster using Ansible and Helm for an application that compares the similarity of legal documents uploaded by the user.

# Kubernetes Cluster Setup and Document Similarity Web Application

This project demonstrates the setup of a Kubernetes cluster using Ansible and Helm to deploy a web application that compares the similarity of legal documents using embeddings created by a third-party API. The application is designed to be scalable and can be extended to use local GPU-enabled machines for future versions.

## Repository Structure
```
k8s-cluster-setup/
├── ansible/
│   ├── k8s-setup.yml
│   ├── inventory.ini
│   ├── roles/
│   │   ├── common/
│   │   │   ├── tasks/
│   │   │   │   └── main.yml
│   │   ├── master/
│   │   │   ├── tasks/
│   │   │   │   └── main.yml
│   │   └── worker/
│   │       ├── tasks/
│   │       │   └── main.yml
├── helm/
│   ├── Chart.yaml
│   ├── templates/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   └── values.yaml
├── README.md
```
## Key Components

### Ansible

#### `ansible/playbook.yml`
Ansible playbook to set up a Kubernetes cluster. This playbook automates the provisioning and configuration of master and worker nodes.

#### `ansible/inventory.ini`
Inventory file containing details of the master and worker nodes. This file specifies the IP addresses and SSH details for the nodes.


#### common/tasks/main.yaml
This file contains tasks that are common to both master and worker nodes. These tasks usually involve installing necessary dependencies, setting up networking, and configuring system settings that are required on all nodes in the Kubernetes cluster.

#### master/tasks/main.yaml
This file contains tasks specific to setting up the master node in the Kubernetes cluster. This includes initializing the Kubernetes master, setting up the API server, controller manager, scheduler, and generating the `kubeadm` join command for worker nodes to join the cluster.

#### worker/tasks/main.yaml
file contains tasks specific to setting up the worker nodes in the Kubernetes cluster. This includes joining the worker nodes to the Kubernetes cluster using the `kubeadm join` command generated by the master node.

### Helm

```
dummy-app/
├── Chart.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── serviceaccount.yaml
│   ├── _helpers.tpl
│   └── tests/
│       └── test-connection.yaml
└── values.yaml
```

#### Chart.yaml:
Contains metadata about the Helm chart, such as the name, version, and description.

#### values.yaml:
Defines the default values for the Helm chart. These values can be overridden by users during the installation process. It includes configurations for replica count, image details, service type, ingress settings, resource limits, and autoscaling.

#### templates/deployment.yaml:
Defines the Kubernetes Deployment of the dummy application. This file specifies the deployment configuration for the web application, including the number of replicas, container image, ports, probes, and resource limits.

#### templates/service.yaml:
Defines the Kubernetes Service resource. This file specifies the configuration for exposing the web application internally within the cluster.

#### templates/ingress.yaml:
Defines the Kubernetes Ingress resource. This file specifies the ingress configuration for managing external access to the web application.

#### templates/hpa.yaml:
Defines the Kubernetes Horizontal Pod Autoscaler (HPA) resource. This file specifies the autoscaling configuration for the deployment, based on CPU utilization or other metrics.

#### templates/serviceaccount.yaml:
Defines the Kubernetes ServiceAccount resource. This file specifies the service account configuration for the deployment, including any necessary annotations and permissions.

#### templates/_helpers.tpl:
Contains template helper functions that are used in other template files. These functions help generate names, labels, and other template values.

#### templates/tests/test-connection.yaml:
Defines a Kubernetes Pod resource for testing the connection to the deployed service. This file specifies a test pod that uses the `busybox` image to verify that the service is accessible.

## Instructions

####  Set Up Kubernetes Cluster
Use the Ansible playbook to provision the Kubernetes cluster.
```
ansible-playbook -i ansible/inventory.ini ansible/playbook.yml
```
#### Install Helm
Install Helm on local machine.
```
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```
#### Deploy Application
Use the Helm Chart to deploy the web application into the Kubernetes cluster.
```
helm install myapp ./dummy-app
```

#### Verify Deployment
Ensure that the application is running and accessible through the configured Ingress using `kubectl`
```
kubectl get all -l app.kubernetes.io/name=myapp
```
#### Run Connection Test
Run the Helm test to verify that the service is accessible.
```
helm test myapp
```
#### Uninstall the Helm Chart
If you want to uninstall the Helm chart, use the
```
helm uninstall
```
#### Future Work
Integrate GPU Workloads: Extend the current setup to deploy containers using GPUs for creating embeddings.
Performance Optimization: Monitor the application's performance and optimize resource allocation.
Security Enhancements: Implement network policies and role-based access control (RBAC) for enhanced security.

## Testing the Setup
To verify the setup, we can:
1)Check the status of the Kubernetes cluster.
2)Ensure that the dummy application is running.
3)Replace the dummy application with the actual application and test the functionality of comparing legal document similarities.

## Developing the Actual Application:
To develop the actual application that will compare the similarity of legal documents, the application should include:
1)A web frontend for users to upload documents.
2)A backend service that processes the uploaded documents, generates embeddings using a third-party API (like OpenAI), and computes cosine similarity between the base document and other documents.

Lets assume we have the following componets:
#### Frontend: 
A simple web interface to upload documents.
#### Backend: 
A service that handles document uploads, calls the embedding API, and computes similarities.

The we can `Containerize the Application`:
Create Dockerfiles for both the frontend and backend services:
For eg:
```
#frontend/Dockerfile

FROM node:14

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000
CMD ["npm", "start"]
```
```
#backend/Dockerfile

FROM python:3.8

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```
Then we will `Build and Push Docker Images`:
```
# Build frontend image
docker build -t <registry_name>/legal-docs-frontend:latest -f frontend/Dockerfile frontend/

# Build backend image
docker build -t <registry_name>/legal-docs-backend:latest -f backend/Dockerfile backend/

# Push images to registry
docker push <registry_name>/legal-docs-frontend:latest
docker push <registry_name>/legal-docs-backend:latest
```
Then we can `Update our Helm chart` to deploy the actual application instead of the dummy application.

##  Integrate local GPU-enabled machines

#### Install GPU Drivers
Ensure that the GPU drivers (e.g., NVIDIA drivers) are installed on the nodes that will be used for GPU workloads. GPU drivers make it possible for our system and software to use the GPU for accelerating tasks.
####  Install NVIDIA Docker Runtime
Install the NVIDIA Docker runtime on these nodes to enable Docker to use the GPUs.
#### Configure Kubernetes for GPUs
Install the NVIDIA device plugin for Kubernetes, which allows Kubernetes to manage GPU resources.
####  Modify Helm Charts for GPU Support
Update your Helm chart to include GPU resource requests and limits in the `values.yaml` and `deployment.yaml`
#### Update Ansible Playbooks
Ensure that the Ansible playbooks are updated to install GPU drivers, NVIDIA Docker runtime, and configure Kubernetes nodes to use GPUs.
#### Deploying the Application
Deploy the updated Helm chart to our Kubernetes cluster. The pods should now be scheduled on GPU-enabled nodes and have access to GPU resources.
#### Test GPU Utilization 
Deploy a simple GPU workload (e.g., TensorFlow or PyTorch job) to verify that the GPUs are being utilized.
#### Monitor Resource Usage
Use Kubernetes dashboards or monitoring tools like Prometheus and Grafana to monitor GPU resource usage and performance.

### Conclusion
By following these steps, we can extend your Kubernetes architecture to integrate local GPU-enabled machines, enabling efficient and scalable deployment of GPU workloads for creating document embeddings. This setup will support the web application in comparing the similarity of legal documents, leveraging the power of GPUs to handle the computationally intensive tasks involved.
