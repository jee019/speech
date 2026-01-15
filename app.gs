function doGet(e) {
	var params = e.parameter;
	var SpreadSheet = SpreadsheetApp.openById("1w7m_lJl0Pbz8spVzofoNJQpPjypcsQXe0NNwzHuwDX4");
	var Sheet = SpreadSheet.getSheets()[0];
	var LastRow = Sheet.getLastRow();

	// Collect all question answers
	// Supports: eng_nmos_q1, eng_smos_q1, kor_nmos_q1, kor_smos_q1, etc.
	var allQuestions = [];
	
	// Collect all question parameters
	for (var key in params) {
		// Skip non-question parameters
		if (key === 'name' || key === 'thank' || key === 'mail' || key === 'formid' || key === 'type') {
			continue;
		}
		// Collect all question parameters (eng_nmos_q1, eng_smos_q1, kor_nmos_q1, kor_smos_q1, etc.)
		if (key.match(/^(eng_|kor_)?(nmos_|smos_)?q\d+$/)) {
			allQuestions.push({
				key: key,
				value: params[key]
			});
		}
	}
	
	// Sort questions: ENG NMOS -> ENG SMOS -> KOR NMOS -> KOR SMOS
	allQuestions.sort(function(a, b) {
		// Extract prefix (eng_nmos_, eng_smos_, kor_nmos_, kor_smos_)
		var prefixA = a.key.match(/^(eng_|kor_)?(nmos_|smos_)/);
		var prefixB = b.key.match(/^(eng_|kor_)?(nmos_|smos_)/);
		
		var prefixA_str = prefixA ? prefixA[0] : '';
		var prefixB_str = prefixB ? prefixB[0] : '';
		
		// Define sort order
		var order = {
			'eng_nmos_': 1,
			'eng_smos_': 2,
			'kor_nmos_': 3,
			'kor_smos_': 4,
			'nmos_': 5,  // fallback for old format
			'smos_': 6   // fallback for old format
		};
		
		var orderA = order[prefixA_str] || 99;
		var orderB = order[prefixB_str] || 99;
		
		if (orderA !== orderB) {
			return orderA - orderB;
		}
		
		// If same prefix, sort by number
		var numA = parseInt(a.key.replace(/^(eng_|kor_)?(nmos_|smos_)?q/, ''));
		var numB = parseInt(b.key.replace(/^(eng_|kor_)?(nmos_|smos_)?q/, ''));
		return numA - numB;
	});

	// Write basic information
	var newRow = LastRow + 1;
	Sheet.getRange(newRow, 1).setValue(params.name || "");
	
	// Write all question answers dynamically
	// Structure: Name | ENG NMOS | ENG SMOS | (blank) | KOR NMOS | KOR SMOS
	var currentCol = 2; // Start from column 2 (column 1 is name)
	var previousPrefix = '';
	
	for (var i = 0; i < allQuestions.length; i++) {
		var question = allQuestions[i];
		
		// Extract prefix
		var prefixMatch = question.key.match(/^(eng_|kor_)?(nmos_|smos_)/);
		var currentPrefix = prefixMatch ? prefixMatch[0] : '';
		
		// Add blank column when transitioning between sections:
		// ENG NMOS -> ENG SMOS: no blank
		// ENG SMOS -> KOR NMOS: add blank
		// KOR NMOS -> KOR SMOS: no blank
		if (previousPrefix !== '' && currentPrefix !== '' && previousPrefix !== currentPrefix) {
			if ((previousPrefix === 'eng_smos_' && currentPrefix === 'kor_nmos_') ||
			    (previousPrefix.match(/^smos_/) && currentPrefix.match(/^nmos_/))) {
				currentCol++; // Skip one column between major sections
			}
		}
		
		// Write only the value
		Sheet.getRange(newRow, currentCol).setValue(question.value || "");
		currentCol++;
		previousPrefix = currentPrefix;
	}

	var thankMessage = params.thank || "참여해주셔서 감사합니다!";
	return ContentService.createTextOutput("제출 완료\n\n" + thankMessage + "\n\n이제 이 창을 닫으셔도 됩니다.");
}
