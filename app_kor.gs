function doGet(e) {
	var params = e.parameter;
	var SpreadSheet = SpreadsheetApp.openById("YOUR_KOR_SHEET_ID_HERE");  // KOR용 Sheet ID로 변경 필요
	var Sheet = SpreadSheet.getSheets()[0];
	var LastRow = Sheet.getLastRow();

	// Collect all question answers (KOR only: kor_nmos_q1, kor_smos_q1, etc.)
	var allQuestions = [];
	
	// Collect all question parameters
	for (var key in params) {
		// Skip non-question parameters
		if (key === 'name' || key === 'thank' || key === 'mail' || key === 'formid' || key === 'type') {
			continue;
		}
		// Collect only KOR question parameters (kor_nmos_q1, kor_smos_q1, etc.)
		if (key.match(/^kor_(nmos_|smos_)q\d+$/)) {
			allQuestions.push({
				key: key,
				value: params[key]
			});
		}
	}
	
	// Sort questions: KOR NMOS -> KOR SMOS
	allQuestions.sort(function(a, b) {
		// Extract prefix (kor_nmos_, kor_smos_)
		var prefixA = a.key.match(/^kor_(nmos_|smos_)/);
		var prefixB = b.key.match(/^kor_(nmos_|smos_)/);
		
		var prefixA_str = prefixA ? prefixA[0] : '';
		var prefixB_str = prefixB ? prefixB[0] : '';
		
		// Define sort order
		var order = {
			'kor_nmos_': 1,
			'kor_smos_': 2
		};
		
		var orderA = order[prefixA_str] || 99;
		var orderB = order[prefixB_str] || 99;
		
		if (orderA !== orderB) {
			return orderA - orderB;
		}
		
		// If same prefix, sort by number
		var numA = parseInt(a.key.replace(/^kor_(nmos_|smos_)?q/, ''));
		var numB = parseInt(b.key.replace(/^kor_(nmos_|smos_)?q/, ''));
		return numA - numB;
	});

	// Write basic information
	var newRow = LastRow + 1;
	Sheet.getRange(newRow, 1).setValue(params.name || "");
	
	// Write all question answers dynamically
	// Structure: Name | KOR NMOS (60개) | KOR SMOS (60개)
	var currentCol = 2; // Start from column 2 (column 1 is name)
	var previousPrefix = '';
	
	for (var i = 0; i < allQuestions.length; i++) {
		var question = allQuestions[i];
		
		// Extract prefix
		var prefixMatch = question.key.match(/^kor_(nmos_|smos_)/);
		var currentPrefix = prefixMatch ? prefixMatch[0] : '';
		
		// Add blank column when transitioning from NMOS to SMOS
		if (previousPrefix === 'kor_nmos_' && currentPrefix === 'kor_smos_') {
			currentCol++; // Skip one column between NMOS and SMOS
		}
		
		// Write only the value
		Sheet.getRange(newRow, currentCol).setValue(question.value || "");
		currentCol++;
		previousPrefix = currentPrefix;
	}

	var thankMessage = params.thank || "참여해주셔서 감사합니다!";
	return ContentService.createTextOutput("제출 완료\n\n" + thankMessage + "\n\n이제 이 창을 닫으셔도 됩니다.");
}
