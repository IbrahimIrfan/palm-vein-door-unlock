const express = require('express');
const app = express();
const bodyParser = require('body-parser');
const port = 8000;
const fsPath = require('fs-path');

app.all('/', function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "X-Requested-With");
  next();
})

app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

app.post('/', (req, res) => {
	img = req.body.img
	processed = req.body.processed
	fsPath.writeFile('images/image.jpeg', img, "base64", function(err){
    	console.log("File saved to images/");
    });
	fsPath.writeFile('images/processed.jpeg', processed, "base64", function(err){
    	console.log("File saved to images/");
    });
	res.send("done");
});

app.listen(port, () => console.log('listening on port ' + port));
