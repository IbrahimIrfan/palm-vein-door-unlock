const express = require('express')
const app = express()

app.get('/', (req, res) => {
	images = [req.body.img1, req.body.img2]

})

app.listen(8000, () => console.log('Server listening on 8000'))
