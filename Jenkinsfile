pipeline {
    agent any 
    parameters {
        string (
            defaultValue: 'mikolajpokrywka',
            description: 'Kaggle username',
            name: 'KAGGLE_USERNAME',
            trim: false
        )
        password(
            defaultValue: '',
            description: 'Kaggle token',
            name: 'KAGGLE_KEY'
        )
        string (
            defaultValue: '10000',
            description: 'cut data',
            name: 'CUTOFF',
            trim: false
        )
    stages {
        stage('checkout: Check out from version control') {
            steps { 
                    checkout([$class: 'GitSCM', branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 's444463', url: 'https://git.wmi.amu.edu.pl/s444463/ium_444463.git']]])
                }
        }
        stage('bash script') {
            steps {
                withEnv(["KAGGLE_USERNAME=${params.KAGGLE_USERNAME}", "KAGGLE_KEY=${params.KAGGLE_KEY}" ]) {
                sh "./process_data.sh"
                archiveArtifacts artifacts: "data_test.csv, data_dev.csv, data_train.csv"
            }
        }
    }
}
