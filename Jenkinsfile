pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
        GCP_PROJECT = "double-vision-456416-a6"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"
    }

    stages {

        stage('Cloning github to jenkins') {
            steps {
                script {
                    echo 'Cloning github to jenkins.....'
                    checkout scmGit(
                        branches: [[name: '*/main']],
                        extensions: [],
                        userRemoteConfigs: [[
                            credentialsId: 'github-token',
                            url: 'https://github.com/Reshendraraj/MLOPS-classification-project.git'
                        ]]
                    )
                }
            }
        }

        stage('Setting up virtual Environment and Installing dependencies') {
            steps {
                script {
                    echo 'Setting up virtual Environment and Installing dependencies.....'
                    sh """#!/bin/bash
                    python -m venv ${VENV_DIR}
                    source ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install --no-cache-dir -e .
                    """
                }
            }
        }

        stage('Building and pushing docker image to GCR') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'Building and pushing docker image to GCR......'
                        sh """#!/bin/bash
                        export PATH=\$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=\${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project \${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/\${GCP_PROJECT}/ml-project:latest .

                        docker push gcr.io/\${GCP_PROJECT}/ml-project:latest
                        """
                    }
                }
            }
        }


        stage('Deploy to google cloud run') {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    script {
                        echo 'Deploy to google cloud run......'
                        sh """#!/bin/bash
                        export PATH=\$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --key-file=\${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project \${GCP_PROJECT}

                        gcloud run deploy ml-project-service \\
                          --image gcr.io/\${GCP_PROJECT}/ml-project:latest \\
                          --region us-central1 \\
                          --platform managed \\
                          --allow-unauthenticated
                        """
                    }
                }
            }
        }

    }
}