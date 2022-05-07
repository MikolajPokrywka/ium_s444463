pipeline {
  	agent {
		dockerfile true
	}
    parameters {
        string (
            defaultValue: '10',
            description: 'Epochs number',
            name: 'EPOCH',
            trim: false
        )
    }
    stages {
        stage('checkout: Check out from version control') {
            steps { 
                    checkout([$class: 'GitSCM', branches: [[name: '*/master']], extensions: [], userRemoteConfigs: [[credentialsId: 's444463', url: 'https://git.wmi.amu.edu.pl/s444463/ium_444463.git']]])
                }
        }
        stage('bash script') {
            steps {
                withEnv(["EPOCH=${params.EPOCH}"]) {
                            copyArtifacts filter: '*', projectName: 's444463-create-dataset'
                            sh 'python3 ./deepl.py  with "epochs=$EPOCH"'
                            mv _sources/deepl_* source.py
                            archiveArtifacts artifacts: "model, source.py"
                            build job: "s444463-evaluation/master"
                }
            }
        }
    }
    post {
        success {
            emailext body: "Model successfully trained", subject: "Model training 444463", to: "e19191c5.uam.onmicrosoft.com@emea.teams.ms"
        }

        failure {
            emailext body: "Training failure", subject: "Model training 444463", to: "e19191c5.uam.onmicrosoft.com@emea.teams.ms"
        }
    }
}
