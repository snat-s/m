-- LH: Little Helper
local M = {}
local api = vim.api
local fn = vim.fn
local JSON = vim.json

-- Configuration
local config = {
  api_key = "<API KEY HERE>", -- Set your DeepSeek API key here
  model = "deepseek-coder",
  system_message = "You are a helpful coding assistant. Respond ONLY in code blocks.",
}

-- Function to make streaming API request
local function make_streaming_api_request(messages)
  local curl_command = string.format(
    "curl -sN -X POST https://api.deepseek.com/v1/chat/completions " ..
    "-H 'Content-Type: application/json' " ..
    "-H 'Authorization: Bearer %s' " ..
    "-d '{\"model\": \"%s\", \"messages\": %s, \"stream\": true}'",
    config.api_key,
    config.model,
    vim.json.encode(messages)
  )
  local response_buffer = ""
  local is_first_content = true
  local job_id = fn.jobstart(curl_command, {
    on_stdout = function(_, data)
      for _, line in ipairs(data) do
        if line ~= "" then
          response_buffer = response_buffer .. line
          while true do
            local start, finish = string.find(response_buffer, "data: ")
            if not start then break end
            local json_str = string.sub(response_buffer, finish + 1)
            response_buffer = string.sub(response_buffer, finish + 1)
            
            local success, decoded = pcall(vim.json.decode, json_str)
            if success and decoded.choices and decoded.choices[1].delta.content then
              local content = decoded.choices[1].delta.content
              local lines = vim.split(content, "\n", true)
              if is_first_content then
                api.nvim_put({"", lines[1]}, '', true, true)
                is_first_content = false
                if #lines > 1 then
                  api.nvim_put(vim.list_slice(lines, 2), '', true, true)
                end
              else
                api.nvim_put(lines, '', true, true)
              end
            end
          end
        end
      end
    end,
    on_exit = function()
      -- Optionally, you can add a newline at the end of the response
      api.nvim_put({""}, 'l', true, true)
    end
  })
end

-- Function to get visual selection
local function get_visual_selection()
  local start_pos = fn.getpos("'<")
  local end_pos = fn.getpos("'>")
  local start_row, start_col = start_pos[2], start_pos[3]
  local end_row, end_col = end_pos[2], end_pos[3]
  
  if start_row > end_row or (start_row == end_row and start_col > end_col) then
    start_row, end_row = end_row, start_row
    start_col, end_col = end_col, start_col
  end
  
  local lines = api.nvim_buf_get_lines(0, start_row - 1, end_row, false)
  if #lines == 0 then return '' end
  
  lines[1] = string.sub(lines[1], start_col)
  if #lines > 1 then
    lines[#lines] = string.sub(lines[#lines], 1, end_col - 1)
  else
    lines[1] = string.sub(lines[1], 1, end_col - start_col + 1)
  end
  
  return table.concat(lines, '\n')
end

-- Main autocompletion function
function M.autocomplete()
  local selected_text = get_visual_selection()
  -- if you are debugging selection, print this
  -- print("Selected text: " .. selected_text)
  
  if selected_text == '' then
    print("No text selected")
    return
  end
  
  -- Get the end position of the selection
  local end_pos = fn.getpos("'>")
  local end_row = end_pos[2]
  
  -- Get the total number of lines in the buffer
  local total_lines = vim.api.nvim_buf_line_count(0)
  
  -- Add a new line after the selection, or at the end of the file if necessary
  if end_row == total_lines then
    vim.api.nvim_buf_set_lines(0, -1, -1, false, {""})
  else
    vim.api.nvim_buf_set_lines(0, end_row, end_row, false, {""})
  end
  
  -- Move the cursor to the newly added line
  vim.api.nvim_win_set_cursor(0, {end_row + 1, 0})
  
  local messages = {
    {role = "system", content = config.system_message},
    {role = "user", content = selected_text}
  }
  
  make_streaming_api_request(messages)
end


-- Setup function
function M.setup(user_config)
  config = vim.tbl_deep_extend("force", config, user_config or {})
  -- Set up the key mapping
  vim.api.nvim_set_keymap('v', '<leader>s', ':<C-u>lua require("lh").autocomplete()<CR>', {noremap = true, silent = true})
end

return M
