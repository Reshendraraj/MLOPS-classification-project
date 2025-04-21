pipeline{
    agent any

    stages{
        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning github repo to Jenkins.....'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/Reshendraraj/MLOPS-classification-project.git']])
                }
            }
        }
    }
}