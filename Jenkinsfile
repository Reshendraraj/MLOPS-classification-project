pipeline{
    agent any

    environment{
        VENV_DIR ='venv'
    }

    stages{
        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning github repo to Jenkins.....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Reshendraraj/MLOPS-classification-project.git']])
                }
            }
        }

         stage('Setting up Virtual Environment and Installing dependancies'){
            steps{
                script{
                    echo 'Setting up Virtual Environment and Installing dependancies.....'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR/bin/activate}
                    pip insatll --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
    }
}