var socket = io.connect('http://' + document.domain + ':' + location.port + '/message');

$(document).ready(function(){

	socket.on('state', function(msg) {
		// show/hide buttons

	});
	socket.on('response', function(msg) {
	    $('#caption').text("")
		$('#caption').text(msg.caption);
	});
	socket.on('log', function(msg) {
		$('#log').val($('#log').val() +"\n"+ msg.data );
	});



});

function start(){
	socket.emit('start', {dataset: $('#dataset').val(), nt:$('#nt').val() , nv: $('#nv').val()});

	return true;

}

function stop(){
	socket.emit('stop');

	return true;

}

function resume(){
	socket.emit('resume');

	return true;

}

function caption(){

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


