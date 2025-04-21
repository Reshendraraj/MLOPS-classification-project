pipeline{
    agent any

    environment{
        VENV_DIR='venv'
        GCP_PROJECT ="double-vision-456416-a6"
        GCLOUD_PATH = "/var/jenkins_home/google-cloud-sdk/bin"

    }


    stages{
        stage('Cloning github to jenkins'){
            steps{
                script{
                    echo 'Cloning github to jenkins.....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Reshendraraj/MLOPS-classification-project.git']])
                }
            }
        }

        stage('Setting up virtual Environment and Insatlling dependancies'){
            steps{
                script{
                    echo 'Setting up virtual Environment and Insatlling dependancies.....'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage('Building and pushing docker image to GCR'){
            steps{
                withCredentials([file(credentialsId : 'gcp-key',  variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Building and pushing docker image to GCR......'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        gcloud auth activate-service-account --keyfile=${GOOGLE_APPLICATION_CREDENTIALS}

                        gcloud config set project${GCP_PROJECT}

                        gcloud auth configure-docker --quiet

                        docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

                        docker push gcr.io/${GCP_PROJECT}/ml-project:latest 
                        '''
                    }
                }
                }
            }
        }
    }
}