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
	if (req.body.type == "image") {
		img = req.body.img;
		fsPath.writeFile('images/' + req.body.name + ".jpg", img, "base64", function(err){
    		console.log("File saved to images/");
    	});
	} else {
		label = req.body.label;
	}
	res.send("done");
});

app.listen(port, () => console.log('listening on port ' + port));
