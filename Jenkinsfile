pipeline {
    agent any 
    stages {
        stage('checkout: Check out from version control') {
            steps {
                    checkout([$class: 'GitSCM', branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 's444463', url: 'https://git.wmi.amu.edu.pl/s444463/ium_444463.git']]])
                }
        }
        stage('bash script') {
            steps {
                sh "./process_data.sh"
            }
        }
    }
}
