package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"strings"
	"sync"
)

// Email represents the original JSON structure from the input file.
type Email struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}

// InnerEmail represents the nested JSON structure stored as a string in the "text" field.
type InnerEmail struct {
	Text string `json:"text"`
}

// CleanedEmail represents the desired output structure.
type CleanedEmail struct {
	Text string `json:"text"`
}

// Regular expressions to extract information even from malformed JSON
var (
	senderRegex    = regexp.MustCompile(`SENDER:\s*([^<\n]+(?:<[^>]+>)?)`)
	receiverRegex  = regexp.MustCompile(`RECEIVER:\s*([^\n]+)`)
	dateRegex      = regexp.MustCompile(`DATE:\s*([^\n]+)`)
	subjectRegex   = regexp.MustCompile(`SUBJECT:\s*([^\n]+)`)
	bodyRegex      = regexp.MustCompile(`BODY\s*(.+)`)
	receiverAltRegex = regexp.MustCompile(`_receiver:\s*([^\n]+)`)
)

// processLine takes a raw JSON line, unmarshals the outer and inner JSON,
// cleans escape characters, and ensures the email text has the necessary headers.
func processLine(line string) (*CleanedEmail, error) {
	var email Email
	// Unmarshal the outer JSON object.
	if err := json.Unmarshal([]byte(line), &email); err != nil {
		return nil, fmt.Errorf("failed to unmarshal outer JSON: %w", err)
	}

	// Try standard JSON unmarshaling first
	var inner InnerEmail
	cleanedText := ""
	
	err := json.Unmarshal([]byte(email.Text), &inner)
	if err == nil {
		// Standard unmarshaling worked
		cleanedText = inner.Text
	} else {
		// If JSON parsing fails, try to extract the content directly
		// Remove the outer quotes and handle JSON escaping
		content := email.Text
		
		// Remove the outermost quotes and braces
		content = strings.TrimPrefix(strings.TrimSuffix(content, "\""), "\"")
		content = strings.TrimPrefix(strings.TrimSuffix(content, "}"), "{")
		
		// Handle text field if present
		if strings.Contains(content, "\"text\":") {
			parts := strings.SplitN(content, "\"text\":", 2)
			if len(parts) > 1 {
				content = parts[1]
				content = strings.TrimPrefix(content, "\"")
				content = strings.TrimSuffix(content, "\"")
			}
		}
		
		// Clean up escape sequences
		cleanedText = content
	}

	// Replace escape sequences with actual characters
	cleanedText = strings.ReplaceAll(cleanedText, "\\n", "\n")
	cleanedText = strings.ReplaceAll(cleanedText, "\\t", "\t")
	cleanedText = strings.ReplaceAll(cleanedText, "\\\"", "\"")
	cleanedText = strings.ReplaceAll(cleanedText, "\\\\", "\\")
	cleanedText = strings.TrimSpace(cleanedText)

	// Replace problematic characters that might be from HTML entities
	cleanedText = strings.ReplaceAll(cleanedText, "\\u003c", "<")
	cleanedText = strings.ReplaceAll(cleanedText, "\\u003e", ">")
	
	// Fix common issues in the email format
	// Convert "_receiver:" to "RECEIVER:"
	cleanedText = strings.ReplaceAll(cleanedText, "_receiver:", "RECEIVER:")
	
	// Ensure "BODY" appears as its own header line if it's embedded in text
	if !strings.Contains(cleanedText, "\nBODY\n") && strings.Contains(cleanedText, "BODY ") {
		cleanedText = strings.ReplaceAll(cleanedText, "BODY ", "BODY\n")
	}

	// If the content doesn't have standard headers, try to reconstruct them
	// from the content using regex
	if !strings.Contains(cleanedText, "SENDER:") &&
		!strings.Contains(cleanedText, "RECEIVER:") &&
		!strings.Contains(cleanedText, "DATE:") &&
		!strings.Contains(cleanedText, "SUBJECT:") {
		
		// Try to extract components using regex
		var formattedEmail strings.Builder
		
		// Extract SENDER
		senderMatch := senderRegex.FindStringSubmatch(cleanedText)
		if len(senderMatch) > 1 {
			formattedEmail.WriteString("SENDER: " + senderMatch[1] + "\n")
		}
		
		// Extract RECEIVER
		receiverMatch := receiverRegex.FindStringSubmatch(cleanedText)
		if len(receiverMatch) > 1 {
			formattedEmail.WriteString("RECEIVER: " + receiverMatch[1] + "\n")
		} else {
			// Try alternate format
			receiverAltMatch := receiverAltRegex.FindStringSubmatch(cleanedText)
			if len(receiverAltMatch) > 1 {
				formattedEmail.WriteString("RECEIVER: " + receiverAltMatch[1] + "\n")
			}
		}
		
		// Extract DATE
		dateMatch := dateRegex.FindStringSubmatch(cleanedText)
		if len(dateMatch) > 1 {
			formattedEmail.WriteString("DATE: " + dateMatch[1] + "\n")
		}
		
		// Extract SUBJECT
		subjectMatch := subjectRegex.FindStringSubmatch(cleanedText)
		if len(subjectMatch) > 1 {
			formattedEmail.WriteString("SUBJECT: " + subjectMatch[1] + "\n")
		}
		
		// Extract BODY
		bodyMatch := bodyRegex.FindStringSubmatch(cleanedText)
		if len(bodyMatch) > 1 {
			formattedEmail.WriteString("BODY\n" + bodyMatch[1])
		} else {
			// If no BODY tag found, use the whole text as body
			formattedEmail.WriteString("BODY\n" + cleanedText)
		}
		
		cleanedText = formattedEmail.String()
	}

	// Check if we have at least some standard headers
	headers := []string{"SENDER:", "RECEIVER:", "DATE:", "SUBJECT:", "BODY"}
	headerFound := false
	for _, header := range headers {
		if strings.Contains(cleanedText, header) {
			headerFound = true
			break
		}
	}

	if !headerFound {
		fmt.Printf("Warning: No standard email headers found in text: %.100s...\n", cleanedText)
	}

	return &CleanedEmail{
		Text: cleanedText,
	}, nil
}

func main() {
	// Open the input file.
	inputFile, err := os.Open("synthetic_emails_output.json")
	if err != nil {
		fmt.Printf("Error opening input file: %v\n", err)
		return
	}
	defer inputFile.Close()

	// Create the output file.
	outputFile, err := os.Create("cleaned_synthetic.json")
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)
		return
	}
	defer outputFile.Close()

	// Create a scanner to read the input file line by line.
	scanner := bufio.NewScanner(inputFile)
	// Set a larger buffer for scanning to handle large lines
	const maxScanTokenSize = 1024 * 1024 // 1MB
	buf := make([]byte, maxScanTokenSize)
	scanner.Buffer(buf, maxScanTokenSize)

	// Create channels for sending lines to process and for receiving processed output.
	linesChan := make(chan string, 100)
	outputChan := make(chan string, 100)
	var wg sync.WaitGroup

	// Determine the number of workers to spawn.
	numWorkers := runtime.NumCPU()
	fmt.Printf("Starting %d workers for concurrent processing\n", numWorkers)

	// Worker function: process each line and send the cleaned JSON string to outputChan.
	worker := func() {
		defer wg.Done()
		for line := range linesChan {
			cleaned, err := processLine(line)
			if err != nil {
				fmt.Printf("Error processing line: %v\n", err)
				continue
			}
			cleanedJSON, err := json.Marshal(cleaned)
			if err != nil {
				fmt.Printf("Error marshaling cleaned email: %v\n", err)
				continue
			}
			outputChan <- string(cleanedJSON)
		}
	}

	// Start the worker pool.
	wg.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go worker()
	}

	// Read the input file in a separate goroutine.
	go func() {
		lineCount := 0
		for scanner.Scan() {
			line := scanner.Text()
			linesChan <- line
			lineCount++
			if lineCount%1000 == 0 {
				fmt.Printf("Processed %d lines so far\n", lineCount)
			}
		}
		
		if err := scanner.Err(); err != nil {
			fmt.Printf("Error reading input file: %v\n", err)
		}
		
		close(linesChan)
		fmt.Printf("Total lines read: %d\n", lineCount)
	}()

	// Once all workers are done, close the output channel.
	go func() {
		wg.Wait()
		close(outputChan)
	}()

	// Write the cleaned emails to the output file.
	writer := bufio.NewWriter(outputFile)
	outputCount := 0
	for cleanedLine := range outputChan {
		_, err := writer.WriteString(cleanedLine + "\n")
		if err != nil {
			fmt.Printf("Error writing to output file: %v\n", err)
			break
		}
		outputCount++
		if outputCount%1000 == 0 {
			fmt.Printf("Wrote %d cleaned emails\n", outputCount)
		}
	}
	
	writer.Flush()
	fmt.Printf("Processing complete. %d cleaned emails written to cleaned_synthetic.json\n", outputCount)
} 