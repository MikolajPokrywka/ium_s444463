pipeline {
  	agent {
		dockerfile true
	}
    parameters {
	    gitParameter branchFilter: 'origin/(.*)', defaultValue: 'master', name: 'BRANCH', type: 'PT_BRANCH'
		buildSelector(
			defaultSelector: lastSuccessful(),
			description: 'From which build copy content',
			name: 'BUILD_SELECTOR'
		)
    }
    stages {
        stage('checkout: Check out from version control') {
            steps { 
                    checkout([$class: 'GitSCM', branches: [[name: ' */master']], extensions: [], userRemoteConfigs: [[credentialsId: 's444463', url: 'https://git.wmi.amu.edu.pl/s444463/ium_444463.git']]])
                }
        }
        stage('bash script') {
            steps {
                    copyArtifacts filter: '*', projectName: 's444463-evaluation/master', optional: true
                    copyArtifacts filter: '*', projectName: 's444356-training/${BRANCH}', selector: buildParameter('BUILD_SELECTOR')
                    copyArtifacts filter: '*', projectName: 's444463-create-dataset'
                    sh 'python3 ./evaluation.py'
                    archiveArtifacts artifacts: "metrics.txt, metrics.png"
            }
        }
    }
    post {
        success {
            emailext body: "Model successfully evaluation", subject: "Model evaluation 444463", to: "e19191c5.uam.onmicrosoft.com@emea.teams.ms"
        }

        failure {
            emailext body: "evaluation failure", subject: "Model evaluation 444463", to: "e19191c5.uam.onmicrosoft.com@emea.teams.ms"
        }
    }
}
