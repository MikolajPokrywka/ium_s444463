pipeline {
  	agent {
		dockerfile true
	}
 parameters {
    buildSelector(
            defaultSelector: lastSuccessful(),
            description: 'Which build to use for copying artifacts for predict',
            name: 'BUILD_SELECTOR')
            string(
                defaultValue: '{\\"inputs\\": [400.0]}',
                description: '',
                name: 'EXAMPLE',
                trim: true
            )
        }
    
    stages {
        stage('Script') {
            steps {
                copyArtifacts projectName: 's444409-training/main', selector: buildParameter('BUILD_SELECTOR')
                sh "python3 predict_other.py ${params.EXAMPLE}"
            }
        }
    }
}