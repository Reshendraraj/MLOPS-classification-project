pipeline{
    agent any

    environment{
        VENV_DIR='venv'

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
    }
}