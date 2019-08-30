var socket = io.connect('http://' + document.domain + ':' + location.port + '/message');


$(document).ready(function(){

	socket.on('state', function(msg) {
		// show/hide buttons

		start='<button type="submit" id="startbutton" onclick="starttraining()" class="btn btn-form button">START TRAINING</button>'
		stop='<button type="submit" id="stopbutton" onclick="stoptraining()" class="btn btn-form button">STOP TRAINING</button>'
		resume='<button type="submit" id="resumebutton" onclick="resumetraining()" class="btn btn-form button">RESUME TRAINING</button>'

		$('#button_space').empty()

		if(msg.running){
				$('#button_space').empty()

			$('#button_space').append(stop)

		}
		else{
			$('#button_space').append(start)

if(msg.resume){

			$('#button_space').append(resume)

			        }
}

	});



});

	socket.on('response', function(msg) {
		$('#caption').text("")
		$('#caption').text(msg.caption);
	});
	socket.on('log', function(msg) {
		$('#log').val($('#log').val() + msg.data +"\n");
        $('#log').scrollTop($('#log')[0].scrollHeight);

	});




function starttraining(){
	socket.emit('start', {dataset: $('#dataset').val(), nt:$('#nt').val() , nv: $('#nv').val(),  ne: $('#ne').val()});

	return true;

}

function stoptraining(){
	socket.emit('stop');

	return true;

}

function resumetraining(){
	socket.emit('resume');

	return true;

}

function caption(){

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

			var input = document.getElementById("userimage");
			var fReader = new FileReader();
			fReader.readAsDataURL(input.files[0]);
			fReader.onloadend = function(event){
				var img = document.getElementById("im");
				img.src = event.target.result;
			}

			filename = $('#userimage').val().split('\\').reverse()[0];
			socket.emit('caption', {filename: filename});
			$('#im').attr('src',$('#userimage').val());


		}
	});



	return true;

}


