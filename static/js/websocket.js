var socket = io.connect('http://' + document.domain + ':' + location.port + '/message');
var socketLog = io.connect('http://' + document.domain + ':' + location.port + '/log');
var socketTest = io.connect('http://' + document.domain + ':' + location.port + '/test');


$(document).ready(function(){

	socket.on('state', function(msg) {
		// show/hide buttons

		start='<button type="submit" id="startbutton" onclick="startTraining()" class="btn btn-form button">START TRAINING</button>'
		stop='<button type="submit" id="stopbutton" onclick="stopTraining()" class="btn btn-form button">STOP TRAINING</button>'
		resume='<button type="submit" id="resumebutton" onclick="resumeTraining()" class="btn btn-form button">RESUME TRAINING</button>'

		$('#buttonspace').empty()

		if(msg.running){
			$('#buttonspace').empty()

			$('#buttonspace').append(stop)

		}
		else{
			$('#buttonspace').append(start)

			if(msg.resume){

				$('#buttonspace').append(resume)

			}
		}

		if(msg.evalrun){

			$('#startbutton').prop("disabled",true);
			$('#evalbutton').prop("disabled",true);
			$('#stopbutton').prop("disabled",true);
			$('#resumebutton').prop("disabled",true);
			$('#testcaptionbutton').prop("disabled",true);
			$('#captionbutton').prop("disabled",true);

		}
		else{
			$('#startbutton').prop("disabled",false);
			$('#evalbutton').prop("disabled",false);
			$('#stopbutton').prop("disabled",false);
			$('#resumebutton').prop("disabled",false);
			$('#testcaptionbutton').prop("disabled",false);
			$('#captionbutton').prop("disabled",false);
		}

	});

	socketTest.emit('get', {dataset: $('#testdataset').val()});


});




socketTest.on('images', function(msg) {

	$('#testimages').empty()
	for (var i = 0; i < msg.images.length; i++) {


		$('#testimages').append('<option value="'+msg.images[i]+'">'+msg.images[i]+'</option>')

	}
});


socketTest.on('response', function(msg) {
	$('#testcaptiondiv').text("");


	$('#testcaptiondiv').append("<h5>GENERATED CAPTION:</h5>");



	$('#testcaptiondiv').append("<p>"+msg.caption+"</p>");



	$('#originalscaptiondiv').text("");




	if(msg.originalcaptions){
		$('#originalscaptiondiv').text("ORIGINAL CAPTIONS:");


		$('#originalscaptiondiv').append('<ol id="originalscaption"></ol>');
		for (var i = 0; i < msg.originalcaptions.length; i++) {


			$('#originalscaption').append("<li>"+msg.originalcaptions[i]+ " ("+msg.bleuscores[i]+")</li>");

		}
	}

});


socket.on('evaluation', function(msg) {

	$('#evaluationdiv').text("");
	$('#evaluationdiv').append("<h5>EVALUATION:</h5>");


	for (var i = 0; i < msg.evaluation.length; i++) {

		$('#evaluationdiv').append("<p>"+msg.evaluation[i]+"</p>");
	}


	$('#startbutton').prop("disabled",false);
	$('#evalbutton').prop("disabled",false);
	$('#stopbutton').prop("disabled",false);
	$('#resumebutton').prop("disabled",false);
	$('#testcaptionbutton').prop("disabled",false);
	$('#captionbutton').prop("disabled",false);


});

socket.on('response', function(msg) {

	$('#captiondiv').text("");
	$('#captiondiv').append("<h5>GENERATED CAPTION:</h5>");



	$('#captiondiv').append("<p>"+msg.caption+"</p>");


});


socketLog.on('log', function(msg) {
	$('#log').val($('#log').val() + msg.data +"\n");
	$('#log').scrollTop($('#log')[0].scrollHeight);

});



function getTestImages(){

	socketTest.emit('get', {dataset: $('#testdataset').val()});

}



function startTraining(){
	socket.emit('start', {dataset: $('#dataset').val(), nt:$('#nt').val() , nv: $('#nv').val(),  ne: $('#ne').val()});

	return true;

}

function stopTraining(){
	socket.emit('stop');

	return true;

}

function resumeTraining(){
	socket.emit('resume');

	return true;

}


function evaluatemodel(){


	socket.emit('eval', {n:$('#evn').val()});


	$('#startbutton').prop("disabled",true);
	$('#evalbutton').prop("disabled",true);
	$('#stopbutton').prop("disabled",true);
	$('#resumebutton').prop("disabled",true);
	$('#testcaptionbutton').prop("disabled",true);
	$('#captionbutton').prop("disabled",true);


}


function testCaption(){

	socketTest.emit('caption', {dataset: $('#testdataset').val(), filename:$('#testimages').val() });
	var img = document.getElementById("testim");

	if($('#dataset').val() === 'flickr'){

		img.src = "https://raw.githubusercontent.com/luca-ant/WhatsSee_dataset/master/images/"+$('#testimages').val()

	}
	else if($('#dataset').val() ==='coco'){
		img.src = "http://images.cocodataset.org/train2017/"+$('#testimages').val()

	}
	return true;


}
function genCaption(){

	if ( $('#userimage').val() === ''){
		return false
	}
	var formData = new FormData();
	formData.append('imagefile', $('#userimage')[0].files[0]);
	filename = $('#userimage').val().split('\\').reverse()[0];
	formData.append('filename', filename);
	$.ajax({
		url: 'image',
		type: 'POST',
		data: formData,
		async: false,
		cache: false,
		contentType: false,
		enctype: 'multipart/form-data',
		processData: false,
		success: function (response) {
			if(response==="Image received!"){
				filename = $('#userimage').val().split('\\').reverse()[0];
				socket.emit('caption', {filename: filename});
				$('#im').attr('src',$('#userimage').val());

				var input = document.getElementById("userimage");
				var fReader = new FileReader();
				fReader.readAsDataURL(input.files[0]);
				fReader.onloadend = function(event){
					var img = document.getElementById("im");
					img.src = event.target.result;
				}

			}
			else{
				var img = document.getElementById("im");
				img.src = "";
				$('#caption').text("")
				alert(response)

			}


		}
	});



	return true;

}


